#ifndef NODE_H
#define NODE_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/float64.hpp>
#include "centerpoint.h"
#include "common.h"
#include <chrono>

class CenterPointNode : public rclcpp::Node
{
public:
    CenterPointNode();
    ~CenterPointNode();

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr input_pointcloud_msg);
    void publishDetected3DArray(const std_msgs::msg::Header &header, std::vector<Bndbox> boxes);

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr detection_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr processing_time_pub_;
    CenterPoint centerpoint;
    Params param;
    cudaStream_t stream;
    float *d_points;

    // time debugging
    std::chrono::time_point<std::chrono::steady_clock> t_start_;
    std::chrono::time_point<std::chrono::steady_clock> t_end_;
};

#endif // NODE_H