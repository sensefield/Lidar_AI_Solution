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

std::string Model_File = "../model/rpn_centerhead_sim.plan";
std::string Save_Dir   = "../data/prediction/";

void GetDeviceInfo()
{
    cudaDeviceProp prop;

    int count = 0;
    cudaGetDeviceCount(&count);
    printf("\nGPU has cuda devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        printf("----device id: %d info----\n", i);
        printf("  GPU : %s \n", prop.name);
        printf("  Capbility: %d.%d\n", prop.major, prop.minor);
        printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
        printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
        printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
        printf("  warp size: %d\n", prop.warpSize);
        printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
        printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    printf("\n");
}

bool hasEnding(std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

int getFolderFile(const char *path, std::vector<std::string>& files, const char *suffix = ".bin")
{
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(path)) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            std::string file = ent->d_name;
            if(hasEnding(file, suffix)){
                files.push_back(file.substr(0, file.length()-4));
            }
        }
        closedir(dir);
    } else {
        printf("No such folder: %s.", path);
        exit(EXIT_FAILURE);
    }
    return EXIT_SUCCESS;
}

int loadData(const char *file, void **data, unsigned int *length)
{
    std::fstream dataFile(file, std::ifstream::in);

    if (!dataFile.is_open()) {
        std::cout << "Can't open files: "<< file<<std::endl;
        return -1;
    }

    unsigned int len = 0;
    dataFile.seekg (0, dataFile.end);
    len = dataFile.tellg();
    dataFile.seekg (0, dataFile.beg);

    char *buffer = new char[len];
    if (buffer==NULL) {
        std::cout << "Can't malloc buffer."<<std::endl;
        dataFile.close();
        exit(EXIT_FAILURE);
    }

    dataFile.read(buffer, len);
    dataFile.close();

    *data = (void*)buffer;
    *length = len;
    return 0;  
}

std::unordered_set<int> hashSet;
void SaveBoxPred(std::vector<Bndbox> boxes, std::string file_name)
{
    std::ofstream ofs;
    ofs.open(file_name, std::ios::out);
    ofs.setf(std::ios::fixed, std::ios::floatfield);
    ofs.precision(5);
    if (ofs.is_open()) {
        for (const auto box : boxes) {
            if (std::isnan(box.x)) {
                continue;
            }
            hashSet.insert(box.id);
            ofs << box.x << " ";
            ofs << box.y << " ";
            ofs << box.z << " ";
            ofs << box.w << " ";
            ofs << box.l << " ";
            ofs << box.h << " ";
            ofs << box.vx << " ";
            ofs << box.vy << " ";
            ofs << box.rt << " ";
            ofs << box.id << " ";
            ofs << box.score << " ";
            ofs << "\n";
        }
    }
    else {
      std::cerr << "Output file cannot be opened!" << std::endl;
    }
    ofs.close();
    std::cout << "Saved prediction in: " << file_name << std::endl;
    
    for (const int& element : hashSet) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    return;
}

static bool startswith(const char *s, const char *with, const char **last)
{
    while (*s++ == *with++)
    {
        if (*s == 0 || *with == 0)
            break;
    }
    if (*with == 0)
        *last = s + 1;
    return *with == 0;
}

static void help()
{
    printf(
        "Usage: \n"
        "    ./centerpoint_infer ../data/test/\n"
        "    Run centerpoint(voxelnet) inference with data under ../data/test/\n"
        "    Optional: --verbose, enable verbose log level\n"
    );
    exit(EXIT_SUCCESS);
}

CenterPointNode::CenterPointNode() : Node("centerpoint"), centerpoint(Model_File, false) {
    RCLCPP_INFO(this-> get_logger(), "Node has been started.");

    pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar", rclcpp::SensorDataQoS{}.keep_last(1),
        std::bind(&CenterPointNode::pointCloudCallback, this, std::placeholders::_1));

    Params params;
    cudaStream_t stream = NULL;
    checkCudaErrors(cudaStreamCreate(&stream));

    centerpoint.prepare();
}

CenterPointNode::~CenterPointNode() {
    centerpoint.perf_report();
    checkCudaErrors(cudaFree(d_points));
    checkCudaErrors(cudaStreamDestroy(stream));
}

void CenterPointNode::pointCloudCallback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr input_pointcloud_msg)
{
    unsigned int length = input_pointcloud_msg->data.size();
    size_t points_num = input_pointcloud_msg->height * input_pointcloud_msg->width;

    float *d_points = nullptr;
    checkCudaErrors(cudaMalloc((void **)&d_points, length));
    checkCudaErrors(cudaMemcpy(d_points, input_pointcloud_msg->data.data(), length, cudaMemcpyHostToDevice));

    centerpoint.doinfer((void *)d_points, points_num, stream);

    static int file_index = 1;
    std::string save_file_name = Save_Dir + std::to_string(file_index) + ".txt";
    SaveBoxPred(centerpoint.nms_pred_, save_file_name);

    RCLCPP_INFO(this->get_logger(), "Saved prediction in: %s", save_file_name.c_str());
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