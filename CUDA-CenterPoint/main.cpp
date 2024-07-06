/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <rclcpp/rclcpp.hpp>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <fstream>
#include <memory>
#include <chrono>
#include <dirent.h>

#include "common.h"
#include "node.h"

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/bounding_box3_d.hpp>
#include <vision_msgs/msg/detection3_d.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>

std::string Model_File = "../model/rpn_centerhead_sim.plan";
std::string Save_Dir = "../data/prediction/";

void GetDeviceInfo()
{
    cudaDeviceProp prop;

    int count = 0;
    cudaGetDeviceCount(&count);
    printf("\nGPU has cuda devices: %d\n", count);
    for (int i = 0; i < count; ++i)
    {
        cudaGetDeviceProperties(&prop, i);
        printf("----device id: %d info----\n", i);
        printf("  GPU : %s \n", prop.name);
        printf("  Capbility: %d.%d\n", prop.major, prop.minor);
        printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
        printf("  Const memory: %luKB\n", prop.totalConstMem >> 10);
        printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
        printf("  warp size: %d\n", prop.warpSize);
        printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
        printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    printf("\n");
}

CenterPointNode::CenterPointNode() : Node("centerpoint"), centerpoint(Model_File, false)
{
    RCLCPP_INFO(this->get_logger(), "Node has been started.");

    pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar", rclcpp::SensorDataQoS{}.keep_last(1),
        std::bind(&CenterPointNode::pointCloudCallback, this, std::placeholders::_1));

    detection_pub_ = this->create_publisher<vision_msgs::msg::Detection3DArray>("detections", 10);

    // Params param;
    cudaStream_t stream = NULL;
    checkCudaErrors(cudaStreamCreate(&stream));

    centerpoint.prepare();
}

CenterPointNode::~CenterPointNode()
{
    centerpoint.perf_report();
    checkCudaErrors(cudaFree(d_points));
    checkCudaErrors(cudaStreamDestroy(stream));
}

void CenterPointNode::pointCloudCallback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr input_pointcloud_msg)
{
    size_t points_num = input_pointcloud_msg->height * input_pointcloud_msg->width;

    // x, y, z, intensity, line のみを使用する
    std::vector<float> processed_points;
    processed_points.reserve(points_num * 5); // x, y, z, intensity, line の5要素

    for (size_t i = 0; i < points_num; ++i)
    {
        float x, y, z, intensity, tag, line;
        memcpy(&x, &input_pointcloud_msg->data[i * input_pointcloud_msg->point_step + input_pointcloud_msg->fields[0].offset], sizeof(float));
        memcpy(&y, &input_pointcloud_msg->data[i * input_pointcloud_msg->point_step + input_pointcloud_msg->fields[1].offset], sizeof(float));
        memcpy(&z, &input_pointcloud_msg->data[i * input_pointcloud_msg->point_step + input_pointcloud_msg->fields[2].offset], sizeof(float));
        memcpy(&intensity, &input_pointcloud_msg->data[i * input_pointcloud_msg->point_step + input_pointcloud_msg->fields[3].offset], sizeof(float));
        memcpy(&tag, &input_pointcloud_msg->data[i * input_pointcloud_msg->point_step + input_pointcloud_msg->fields[4].offset], sizeof(uint8_t));
        memcpy(&line, &input_pointcloud_msg->data[i * input_pointcloud_msg->point_step + input_pointcloud_msg->fields[5].offset], sizeof(uint8_t));
        // memcpy(&line, &input_pointcloud_msg->data[i * input_pointcloud_msg->point_step + input_pointcloud_msg->fields[5].offset], sizeof(float));
        processed_points.push_back(x);
        processed_points.push_back(y);
        processed_points.push_back(z);
        processed_points.push_back(intensity);
        processed_points.push_back(line);
        // processed_points.push_back(static_cast<float>(line));
    }
    float *d_points = nullptr;
    size_t processed_points_size = processed_points.size() * sizeof(float);
    checkCudaErrors(cudaMalloc((void **)&d_points, processed_points_size));
    checkCudaErrors(cudaMemcpy(d_points, processed_points.data(), processed_points_size, cudaMemcpyHostToDevice));

    centerpoint.doinfer((void *)d_points, points_num, stream);

    publishDetected3DArray(input_pointcloud_msg->header, centerpoint.nms_pred_);
}

void CenterPointNode::publishDetected3DArray(const std_msgs::msg::Header &header, std::vector<Bndbox> boxes)
{
    auto detection_array_msg = vision_msgs::msg::Detection3DArray();
    detection_array_msg.header = header;
    for (const auto &box : boxes)
    {
        if (std::isnan(box.x))
        {
            continue;
        }
        auto detection_msg = vision_msgs::msg::Detection3D();
        detection_msg.header = header;

        auto hypothesis_with_pose = vision_msgs::msg::ObjectHypothesisWithPose();
        hypothesis_with_pose.hypothesis.class_id = param.class_name[box.id];
        hypothesis_with_pose.hypothesis.score = box.score;
        detection_msg.results.push_back(hypothesis_with_pose);

        auto bbox = vision_msgs::msg::BoundingBox3D();
        bbox.center.position.x = box.x;
        bbox.center.position.y = box.y;
        bbox.center.position.z = box.z;
        bbox.size.x = box.w;
        bbox.size.y = box.l;
        bbox.size.z = box.h;
        detection_msg.bbox = bbox;
        detection_array_msg.detections.push_back(detection_msg);
    }
    detection_pub_->publish(detection_array_msg);
}

int main(int argc, const char **argv)
{
    GetDeviceInfo();

    rclcpp::init(argc, argv);

    auto node = std::make_shared<CenterPointNode>();

    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}