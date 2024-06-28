#ifndef NODE_H
#define NODE_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include "centerpoint.h"
#include "common.h"

class CenterPointNode : public rclcpp::Node {
public:
    CenterPointNode();
    ~CenterPointNode();

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr input_pointcloud_msg);
    
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    CenterPoint centerpoint;
    Params param;
    cudaStream_t stream;
    float *d_points;
};

#endif // NODE_H