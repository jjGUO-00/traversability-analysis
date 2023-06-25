#ifndef _UTILITY_TM_H_
#define _UTILITY_TM_H_

#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <interactive_markers/interactive_marker_server.h>

#include <nav_core/base_global_planner.h>
#include <costmap_2d/costmap_2d_ros.h>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

#include <opencv/cv.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/range_image/range_image.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>

#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <pcl_ros/transforms.h>

#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <queue>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator> 
#include <sstream>
#include <string>
#include <array> // c++11
#include <thread> // c++11
#include <mutex> // c++11
#include <chrono> // c++11

#include <Eigen/Core>

#include "marker/Marker.h"
#include "marker/MarkerArray.h"

#include "planner/kdtree.h"
#include "planner/cubic_spline_interpolator.h"

#include "elevation_msgs/OccupancyElevation.h"

using namespace std;

// X,Y,Z, Intensity, Ring, Label
// struct PointXYZIRL
// {
//     PCL_ADD_POINT4D
//     PCL_ADD_INTENSITY;
//     uint16_t ring;
//     int label;
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
// } EIGEN_ALIGN16;

// POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRL,  
//                                    (float, x, x) (float, y, y)
//                                    (float, z, z) (float, intensity, intensity)
//                                    (uint16_t, ring, ring) (int, label, label)                                
// )
struct PointXYZIRL
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float ring;
    float label;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRL,  
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, ring, ring) (float, label, label)                                
)

// X,Y,Z, Intensity, Label (部分点云无ring信息)
// struct PointXYZIL
// {
//     PCL_ADD_POINT4D
//     PCL_ADD_INTENSITY;
//     int label;
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
// } EIGEN_ALIGN16;

// POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIL,  
//                                    (float, x, x) (float, y, y)
//                                    (float, z, z) (float, intensity, intensity)
//                                    (int, label, label)
// )

struct PointXYZIL
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float label;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIL,  
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, label, label)
)

// 来个pointcloud type PointXYZI(Intensity)O(Occupancy)L(Label)

typedef pcl::PointXYZI  PointType;
typedef struct kdtree kdtree_t;
typedef struct kdres kdres_t;

// Point Cloud Type
extern const bool useCloudRing = false;

// Environment
extern const bool urbanMapping = false;

// rosbag
extern const bool CorrierRosbag = true;

// VLP-16
extern const int N_SCAN = 16;
extern const int Horizon_SCAN = 1800;
// extern const int Horizon_SCAN = 1200;
extern const float sensorMinimumRange = 1.0;
extern const float ang_res_x = 0.2;
extern const float ang_res_y = 2.0;
extern const float ang_bottom = 15.0+0.1;
extern const int groundScanInd = 7;
// Map Params
extern const float mapResolution = 0.1; // map resolution
extern const float mapCubeLength = 1.0; // the length of a sub-map (meters)
extern const int mapCubeArrayLength = mapCubeLength / mapResolution; // the grid dimension of a sub-map (mapCubeLength / mapResolution)
extern const int mapArrayLength = 2000 / mapCubeLength; // the sub-map dimension of global map (2000m x 2000m)
extern const int rootCubeIndex = mapArrayLength / 2; // by default, robot is at the center of global map at the beginning
 
// Filter Ring Params
extern const int scanNumCurbFilter = 8;
extern const int scanNumSlopeFilter = 5;
extern const int scanNumMax = std::max(scanNumCurbFilter, scanNumSlopeFilter);
// extern const int scanNumMax = 16;


// Filter Threshold Params
// extern const float sensorRangeLimit = 12; // only keep points with in ... meters  
extern const float sensorRangeLimit = 10;   // 只考虑10m以内的点
// extern const float sensorRangeLimit = 5;   // 只考虑10m以内的点(建离线可通行地图可以调小一点)
extern const float filterHeightLimit = (urbanMapping == true) ? 0.05 : 0.12; // step diff threshold
// extern const float filterHeightLimit = (urbanMapping == true) ? 0.05 : 0.15; // step diff threshold         
extern const float filterAngleLimit = 25; // slope angle threshold 用于斜率滤波
extern const int filterHeightMapArrayLength = sensorRangeLimit * 2 / mapResolution;  //感知范围内的栅格的维度
extern const float intensityLimit = 10.0;
// BGK Prediction Params
extern const bool predictionEnableFlag = true;
extern const float predictionKernalSize = 0.1; // predict elevation within x meters

// Occupancy Params
extern const float p_occupied_when_laser = 0.9;  // 几何判断被占据
extern const float p_occupied_when_no_laser = 0.3; // 几何判断未被占据
extern const float p_occupied_when_semantic = 0.97; // 语义判断被占据
extern const float p_occupied_when_no_semantic = 0.22; // 语义判断未被占据
extern const float large_log_odds = 100;
extern const float max_log_odds_for_belief = 20;


// 2D Map Publish Params
extern const int localMapLength = 10; // length of the local occupancy grid map (meter)
// extern const int localMapLength = 350; // length of the local occupancy grid map (meter) 全局可通行地图
extern const int localMapArrayLength = localMapLength / mapResolution;  //局部地图的维度

// Visualization Params
extern const float visualizationRadius = 50; //可视化的范围
// extern const float visualizationFrequency = 2; // n, skip n scans then publish, n=0, visualize at each scan
extern const float visualizationFrequency = 2; // n, skip n scans then publish, n=0, visualize at each scan  //每2帧可视化一次
// Robot Params
extern const float robotRadius = 0.2;
// extern const float sensorHeight = 0.5;
extern const float sensorHeight = 0.87;
extern const int robotArrayLength = robotRadius*6 / mapResolution; // 机器人所占栅格地图的直径

// Traversability Params
// extern const int traversabilityObserveTimeTh = 10;  
// extern const float traversabilityCalculatingDistance = 8.0;
extern const int traversabilityObserveTimeTh = 1;     //被稳定观测到的次数
extern const int semObserveTime = 1;  // 语义信息被观测到的次数(2次太久了，跟不上机器人运动的速度)
extern const float traversabilityCalculatingDistance = 8.0;   //超过8m不计算和更新
// Planning Cost Params
extern const int NUM_COSTS = 3;
extern const int tmp[] = {2};
extern const std::vector<int> costHierarchy(tmp, tmp+sizeof(tmp)/sizeof(int));// c++11 initialization: costHierarchy{0, 1, 2}

// PRM Planner Settings
extern const bool planningUnknown = true;
extern const float costmapInflationRadius = 0.1;
extern const float neighborSampleRadius  = 0.5;
extern const float neighborConnectHeight = 1.0;
extern const float neighborConnectRadius = 2.0;
extern const float neighborSearchRadius = localMapLength / 2;

struct grid_t;
struct mapCell_t;
struct childMap_t;
struct state_t;
struct neighbor_t;

/*
    栅格信息存储的结构体，主要存储在地图一维序列中的索引gridIndex，在二维cube地图中的索引cubeX和cubeY，在cube中栅格的索引gridX和gridY
    This struct is used to send map from mapping package to prm package
    */
struct grid_t{
    int mapID;
    int cubeX;
    int cubeY;
    int gridX;
    int gridY;
    int gridIndex;
};

/*
    cube (sub-map)中一个栅格的数据定义
    Cell Definition:
    a cell is a member of a grid in a sub-map
    a grid can have several cells in it. 
    a cell represent one height information
    */

struct mapCell_t{

    PointType *xyz; // it's a pointer to the corresponding point in the point cloud of submap

    grid_t grid;

//  float log_odds;
    float geo_log_odds; // geometry lod odds
    float sem_log_odds; // semantic log odds
    float geo_occupancy;
    float sem_occupancy;
    int8_t sem_obs_times;   // count the times that the cell is observed by camera

    int observeTimes;
    float time;
    
    float occupancy, occupancyVar;
    float elevation, elevationVar;
    float step_diff;
    float step_traversability;
    float slope_traversability;
    float roughness_traversability;
    bool dynamic_label;

    mapCell_t(){
        
        // log_odds = 0.5;
        geo_log_odds = 0.5;
        sem_log_odds = 0.5;
        sem_obs_times = 0;

        observeTimes = 0;

        elevation = -FLT_MAX;
        elevationVar = 1e3;
        // 要加其他的field可以在这里加
        // slope和roughness都是基于elevation的，应该不需要每个cell都维护
        // intensity
        // occupancy的模式应该改掉？改成score
        // occupancy = 0; // initialized as unkown
        // treat occupancy as a traversability score
        occupancy = 0; // initialize as traversable
        occupancyVar = 1e3;
        dynamic_label = false;
        
    }

    void updatePoint(){
        xyz->z = elevation;
        xyz->intensity = occupancy;
    }
    void updateElevation(float elevIn, float varIn){
        elevation = elevIn;
        elevationVar = varIn;
        updatePoint();
    }
    void updateOccupancy(float occupIn){
        occupancy = occupIn;
        updatePoint();
    }
};


/*
    Sub-map Definition:
    childMap_t is a small square. We call it "cellArray". 
    It composes the whole map
    */
struct childMap_t{

    vector<vector<mapCell_t*> > cellArray;
    int subInd; //sub-map's index in 1d mapArray
    int indX; // sub-map's x index in 2d array mapArrayInd
    int indY; // sub-map's y index in 2d array mapArrayInd
    float originX; // sub-map's x root coordinate
    float originY; // sub-map's y root coordinate
    pcl::PointCloud<PointType> cloud;

    childMap_t(int id, int indx, int indy){

        subInd = id;
        indX = indx;
        indY = indy;
        originX = (indX - rootCubeIndex) * mapCubeLength - mapCubeLength/2.0;
        originY = (indY - rootCubeIndex) * mapCubeLength - mapCubeLength/2.0;

        // allocate and initialize each cell
        cellArray.resize(mapCubeArrayLength);
        for (int i = 0; i < mapCubeArrayLength; ++i)
            cellArray[i].resize(mapCubeArrayLength);

        for (int i = 0; i < mapCubeArrayLength; ++i)
            for (int j = 0; j < mapCubeArrayLength; ++j)
                cellArray[i][j] = new mapCell_t;
        // allocate point cloud for visualization
        cloud.points.resize(mapCubeArrayLength*mapCubeArrayLength);

        // initialize cell pointer to cloud point
        for (int i = 0; i < mapCubeArrayLength; ++i)
            for (int j = 0; j < mapCubeArrayLength; ++j)
                cellArray[i][j]->xyz = &cloud.points[i + j*mapCubeArrayLength];

        // initialize each point in the point cloud, also each cell
        for (int i = 0; i < mapCubeArrayLength; ++i){
            for (int j = 0; j < mapCubeArrayLength; ++j){
                
                // point cloud initialization
                int index = i + j * mapCubeArrayLength;
                cloud.points[index].x = originX + i * mapResolution;
                cloud.points[index].y = originY + j * mapResolution;
                cloud.points[index].z = std::numeric_limits<float>::quiet_NaN();
                cloud.points[index].intensity = cellArray[i][j]->occupancy;

                // cell position in the array of submap
                cellArray[i][j]->grid.mapID = subInd;
                cellArray[i][j]->grid.cubeX = indX;
                cellArray[i][j]->grid.cubeY = indY;
                cellArray[i][j]->grid.gridX = i;
                cellArray[i][j]->grid.gridY = j;
                cellArray[i][j]->grid.gridIndex = index;
            }
        }
    }
};



/*
    Robot State Defination
    */


struct state_t{
    double x[3]; //  1 - x, 2 - y, 3 - z
    float theta;
    int stateId;
    float cost;
    bool validFlag;
    // # Cost types
    // # 0. obstacle cost
    // # 1. elevation cost
    // # 2. distance cost
    float costsToRoot[NUM_COSTS];
    float costsToParent[NUM_COSTS]; // used in RRT*
    float costsToGo[NUM_COSTS];

    state_t* parentState; // parent for this state in PRM and RRT*
    vector<neighbor_t> neighborList; // PRM adjencency list with edge costs
    vector<state_t*> childList; // RRT*

    // default initialization
    state_t(){
        parentState = NULL;
        for (int i = 0; i < NUM_COSTS; ++i){
            costsToRoot[i] = FLT_MAX;
            costsToParent[i] = FLT_MAX;
            costsToGo[i] = FLT_MAX;
        }
    }
    // use a state input to initialize new state
    
    state_t(state_t* stateIn){
        // pose initialization
        for (int i = 0; i < 3; ++i)
            x[i] = stateIn->x[i];
        theta = stateIn->theta;
        // regular initialization
        parentState = NULL;
        for (int i = 0; i < NUM_COSTS; ++i){
            costsToRoot[i] = FLT_MAX;
            costsToParent[i] = stateIn->costsToParent[i];
        }
    }
};


struct neighbor_t{
    state_t* neighbor;
    float edgeCosts[NUM_COSTS]; // the cost from this state to neighbor
    neighbor_t(){
        neighbor = NULL;
        for (int i = 0; i < NUM_COSTS; ++i)
            edgeCosts[i] = FLT_MAX;
    }
};









////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////      Some Functions    ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
state_t *compareState;
bool isStateExsiting(neighbor_t neighborIn){
    return neighborIn.neighbor == compareState ? true : false;
}

float pointDistance(PointType p1, PointType p2){
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

#endif