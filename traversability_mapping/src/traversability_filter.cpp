#include "utility.h"
#include <pcl/features/normal_3d.h>

class TraversabilityFilter
{

private:
    // ROS node handler
    ros::NodeHandle nh;
    // ROS subscriber
    ros::Subscriber subCloud;
    // ROS publisher
    ros::Publisher pubCloud;
    ros::Publisher pubCloudVisualHiRes;
    ros::Publisher pubCloudVisualLowRes;
    ros::Publisher pubLaserScan;
    // Point Cloud
     
    pcl::PointCloud<PointXYZIL>::Ptr laserCloudRaw;     // raw cloud from /velodyne_points
    pcl::PointCloud<PointXYZIRL>::Ptr laserCloudInRing; // full cloud with x,y,z,intensity,label,rings
     

    pcl::PointCloud<PointType>::Ptr laserCloudIn;        // projected full velodyne cloud, intensity storing range information
    pcl::PointCloud<PointType>::Ptr laserCloudOut;       // filtered and downsampled point cloud
    pcl::PointCloud<PointType>::Ptr laserCloudObstacles; // cloud for saving points that are classified as obstables, convert them to laser scan
    // Transform Listener
    tf::TransformListener listener;
    tf::StampedTransform transform;
    // A few points

     
    PointType nanPoint;
     

    PointType robotPoint;
    PointType localMapOrigin;
    // point cloud saved as N_SCAN * Horizon_SCAN form
    vector<vector<PointType>> laserCloudMatrix;
    // // 定义两个矩阵，存储点云和语义分割得到的障碍物信息
    cv::Mat geoObstacleMatrix; // store geometry obstacle -1 - invalid, 0 - free, 1 - obstacle
    cv::Mat semObstacleMatrix; // store semantic obstacle -1 - invalid, 0 - free, 1 - static obstacle  2 - dynamic obstacle
    cv::Mat rangeMatrix;       // -1 - invalid, >0 - valid range value
     
    cv::Mat intensityMatrix; // store point's intensity in matrix format
     
    cv::Mat labelMatrix; // store point's label in matrix format
    // laser scan message
    sensor_msgs::LaserScan laserScan;
    // for downsample
    float **minHeight;
    float **maxHeight;
    bool **geoObstFlag;   // 200*200局部地图的几何障碍物标记
    int8_t **semObstFlag; // 200*200局部地图的语义障碍物标记
    bool **initFlag;

     
    ros::Time pcMsgTimeStamp;
    string ns;  // namespace param
     
public:
    TraversabilityFilter() : nh("~"), ns("")
    {
        if(!nh.getParam("namespace", ns)){
            ns = "";
        }
        cout<<"namespace is: "<<ns<<endl;
        // subCloud = nh.subscribe<sensor_msgs::PointCloud2>("/full_cloud_info", 5, &TraversabilityFilter::cloudHandler, this);
        subCloud = nh.subscribe<sensor_msgs::PointCloud2>(ns+"/label_cloud", 5, &TraversabilityFilter::cloudHandler, this);
        pubCloud = nh.advertise<sensor_msgs::PointCloud2>(ns+"/filtered_pointcloud", 5);
        pubCloudVisualHiRes = nh.advertise<sensor_msgs::PointCloud2>(ns+"/filtered_pointcloud_visual_high_res", 5);
        pubCloudVisualLowRes = nh.advertise<sensor_msgs::PointCloud2>(ns+"/filtered_pointcloud_visual_low_res", 5);
        pubLaserScan = nh.advertise<sensor_msgs::LaserScan>(ns+"/pointcloud_2_laserscan", 5);

         
        nanPoint.x = std::numeric_limits<float>::quiet_NaN(); // 判断float类型是否有小数点，有返回1
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;
         
        allocateMemory();

        pointcloud2laserscanInitialization();
    }

    void allocateMemory()
    {

        laserCloudRaw.reset(new pcl::PointCloud<PointXYZIL>());
        laserCloudInRing.reset(new pcl::PointCloud<PointXYZIRL>());

        laserCloudIn.reset(new pcl::PointCloud<PointType>());
        laserCloudIn->points.resize(N_SCAN * Horizon_SCAN);
        laserCloudOut.reset(new pcl::PointCloud<PointType>());
        laserCloudObstacles.reset(new pcl::PointCloud<PointType>());

        geoObstacleMatrix = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(-1)); // CV_32S：有符号32位整型
        semObstacleMatrix = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(-1)); // CV_32S：有符号32位整型
        rangeMatrix = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(-1));       // CV_32F是float - 像素可以有0-1.0之间的任何值
        intensityMatrix = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(-1));   // CV_32F是float - 像素可以有0-1.0之间的任何值
        labelMatrix = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(-1));       // 初始值全部设为-1
        laserCloudMatrix.resize(N_SCAN);
        for (int i = 0; i < N_SCAN; ++i)
            laserCloudMatrix[i].resize(Horizon_SCAN);

        initFlag = new bool *[filterHeightMapArrayLength];
        for (int i = 0; i < filterHeightMapArrayLength; ++i)
            initFlag[i] = new bool[filterHeightMapArrayLength];

        geoObstFlag = new bool *[filterHeightMapArrayLength];
        for (int i = 0; i < filterHeightMapArrayLength; ++i)
            geoObstFlag[i] = new bool[filterHeightMapArrayLength];

        semObstFlag = new int8_t *[filterHeightMapArrayLength];
        for (int i = 0; i < filterHeightMapArrayLength; ++i)
            semObstFlag[i] = new int8_t[filterHeightMapArrayLength];

        minHeight = new float *[filterHeightMapArrayLength];
        for (int i = 0; i < filterHeightMapArrayLength; ++i)
            minHeight[i] = new float[filterHeightMapArrayLength];

        maxHeight = new float *[filterHeightMapArrayLength];
        for (int i = 0; i < filterHeightMapArrayLength; ++i)
            maxHeight[i] = new float[filterHeightMapArrayLength];

        resetParameters();
    }

    void resetParameters()
    {
        laserCloudRaw->clear();

        // laserCloudIn->clear();
        laserCloudOut->clear();
        laserCloudObstacles->clear();

        geoObstacleMatrix = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(-1));
        semObstacleMatrix = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(-1));
        rangeMatrix = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(-1));
         
        intensityMatrix = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(-1));
        labelMatrix = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(-1));
         
        for (int i = 0; i < filterHeightMapArrayLength; ++i)
        {
            for (int j = 0; j < filterHeightMapArrayLength; ++j)
            {
                initFlag[i][j] = false;
                geoObstFlag[i][j] = false;
                semObstFlag[i][j] = -1;
            }
        }

        std::fill(laserCloudIn->points.begin(), laserCloudIn->points.end(), nanPoint);
    }

    ~TraversabilityFilter() {}

    // * main callback
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
    {

         
        velodyne2RangeCloud(laserCloudMsg);
         

        extractRawCloud();

        if (transformCloud() == false)
            return; // 将点云进行坐标变换到全局坐标系

        cloud2Matrix();

        applyFilter();

        extractFilteredCloud();

        downsampleCloud();

        predictCloudBGK();

        publishCloud();
         
        // publishLaserScan();
         
        resetParameters();
    }

    void removeNaNFromPointCloud(const pcl::PointCloud<PointXYZIL> &cloud_in,
                                 pcl::PointCloud<PointXYZIL> &cloud_out,
                                 std::vector<int> &index)
    {
        // If the clouds are not the same, prepare the output
        if (&cloud_in != &cloud_out)
        {
            cloud_out.header = cloud_in.header;
            cloud_out.points.resize(cloud_in.points.size());
        }
        // Reserve enough space for the indices
        index.resize(cloud_in.points.size());
        size_t j = 0;

        // If the data is dense, we don't need to check for NaN
        if (cloud_in.is_dense)
        {
            // Simply copy the data
            cloud_out = cloud_in;
            for (j = 0; j < cloud_out.points.size(); ++j)
                index[j] = static_cast<int>(j);
        }
        else
        {
            for (size_t i = 0; i < cloud_in.points.size(); ++i)
            {
                if (!pcl_isfinite(cloud_in.points[i].x) ||
                    !pcl_isfinite(cloud_in.points[i].y) ||
                    !pcl_isfinite(cloud_in.points[i].z))
                    continue;
                cloud_out.points[j] = cloud_in.points[i];
                index[j] = static_cast<int>(i);
                j++;
            }
            if (j != cloud_in.points.size())
            {
                // Resize to the correct size
                cloud_out.points.resize(j);
                index.resize(j);
            }

            cloud_out.height = 1;
            cloud_out.width = static_cast<uint32_t>(j);

            // Removing bad points => dense (note: 'dense' doesn't mean 'organized')
            cloud_out.is_dense = true;
        }
    }

    // * fill laserCloudIn as LeGO-LOAM format from raw velodyne points
    void velodyne2RangeCloud(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
    {

        pcl::fromROSMsg(*laserCloudMsg, *laserCloudRaw); // Tranform point_cloud msg into point cloud.
        std::vector<int> indices;
        removeNaNFromPointCloud(*laserCloudRaw, *laserCloudRaw, indices);
        if (useCloudRing == true)
        {
            pcl::fromROSMsg(*laserCloudMsg, *laserCloudInRing); // 点云中的每个点所在的线数ring信息
            if (laserCloudInRing->is_dense == false)
            { // True if no points are invalid (e.g., have NaN or Inf values).
                ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
                ros::shutdown();
            }
        }
        // save timestamp (for transform)
        pcMsgTimeStamp = laserCloudMsg->header.stamp;
        size_t cloudSize = laserCloudRaw->points.size();
        // cout << "raw cloud size = " << cloudSize << endl;

         
        // PointType thisPoint;
        PointXYZIL thisPoint;
         

        float verticalAngle, horizonAngle, range;
        size_t rowIdn, columnIdn, index;

        for (size_t i = 0; i < cloudSize; i++)
        {
            thisPoint.x = laserCloudRaw->points[i].x;
            thisPoint.y = laserCloudRaw->points[i].y;
            thisPoint.z = laserCloudRaw->points[i].z;
            thisPoint.intensity = laserCloudRaw->points[i].intensity;
            thisPoint.label = laserCloudRaw->points[i].label;
            if (useCloudRing == true)
            {
                rowIdn = int(laserCloudInRing->points[i].ring);
            }
            else
            {
                verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
                rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
            }

            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
            columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
            // range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y);
            if (range < sensorMinimumRange)
                continue;
            index = columnIdn + rowIdn * Horizon_SCAN;
            // laserCloudIn->points[index] = thisPoint;

            PointType tempPoint;
            tempPoint.x = thisPoint.x; // range image 的三维坐标和点云是一样的，只是增加了距离信息，同时利用空间坐标重新计算对应的index
            tempPoint.y = thisPoint.y;
            tempPoint.z = thisPoint.z;
            laserCloudIn->points[index] = tempPoint;
            laserCloudIn->points[index].intensity = range; // 存储的距离信息

            intensityMatrix.at<float>(rowIdn, columnIdn) = thisPoint.intensity;
            labelMatrix.at<int>(rowIdn, columnIdn) = int(thisPoint.label);
        }
    }

    // * extract range information from point cloud
    void extractRawCloud()
    {
        // ROS msg -> PCL cloud
        // This function takes need call of function "velodyne2RangeCloud"
        int nPoints = laserCloudIn->points.size();
        // extract range info
        for (int i = 0; i < N_SCAN; ++i)
        {
            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                int index = j + i * Horizon_SCAN;
                // skip NaN point
                if (laserCloudIn->points[index].intensity == std::numeric_limits<float>::quiet_NaN())
                    continue;
                // save range info
                rangeMatrix.at<float>(i, j) = laserCloudIn->points[index].intensity; // 存储的距离信息
                // reset obstacle status to 0 - free
                geoObstacleMatrix.at<int>(i, j) = 0;
                semObstacleMatrix.at<int>(i, j) = -1;
                

            }
        }
    }

    // * transform point cloud under different coordinate 点云进行变换到全局坐标系
    bool transformCloud()
    {
        // Listen to the TF transform and prepare for point cloud transformation
        try
        {
            listener.lookupTransform(ns+"/map", ns+"/base_link", ros::Time(0), transform);
        } // 从base_link到map，即机器人在map坐标系下的xyz坐标
        // try{listener.lookupTransform("map","base_link", pcMsgTimeStamp, transform); }
        catch (tf::TransformException ex)
        { /*ROS_ERROR("Transfrom Failure.");*/
            return false;
        }

        robotPoint.x = transform.getOrigin().x();
        robotPoint.y = transform.getOrigin().y();
        robotPoint.z = transform.getOrigin().z();

        laserCloudIn->header.frame_id = ns+"/base_link";
        laserCloudIn->header.stamp = 0; // don't use the latest time, we don't have that transform in the queue yet

        pcl::PointCloud<PointType> laserCloudTemp;
        pcl_ros::transformPointCloud(ns+"/map", *laserCloudIn, laserCloudTemp, listener);
        *laserCloudIn = laserCloudTemp;

        return true;
    }

    // * convert point cloud into matrix  将点云转化为矩阵存储的形式
    void cloud2Matrix()
    {
        for (int i = 0; i < N_SCAN; ++i)
        {
            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                int index = j + i * Horizon_SCAN;
                PointType p = laserCloudIn->points[index];
                laserCloudMatrix[i][j] = p; // 16 * 1800
            }
        }
    }

    // * apply multiple geometric filters for obstacle detection
    void applyFilter()
    {
        labelFilter();
        if (urbanMapping == true){

            positiveCurbFilter_v2();
            negativeCurbFilter();
        }
        else{
            negativeCurbFilter();
        }
        slopeFilter(); // 计算局部点云的斜率，进行进一步的地面点的识别
        // intensityFilter();
    }

    //  for Corriere rosbag label
    // void labelFilter()
    // {
    //     for (int i = 0; i < N_SCAN; i++)
    //     {
    //         for (int j = 0; j < Horizon_SCAN; j++)
    //         {
    //             // 跳过 NaN 点
    //             if (rangeMatrix.at<float>(i, j) == -1 || rangeMatrix.at<float>(i + 1, j) == -1)
    //                 continue;
    //             if(labelMatrix.at<int>(i,j) == 1 || labelMatrix.at<int>(i,j) == 4){
    //                 semObstacleMatrix.at<int>(i,j) = 0; // 可通行
    //             } else if(labelMatrix.at<int>(i,j) != -2 && labelMatrix.at<int>(i,j) != 10) { // label is not "flat"
    //                 // ROS_INFO("label: %d",labelMatrix.at<int>(i,j));
    //                 semObstacleMatrix.at<int>(i,j) = 1;   // 静态障碍物
    //             }else if(labelMatrix.at<int>(i,j) == 10){ // 假设10是动态障碍物
    //                 // ROS_INFO("dynamic obstacle label: %d",labelMatrix.at<int>(i,j));
    //                 semObstacleMatrix.at<int>(i,j) = 2;  // 动态障碍物
    //             }
    //         }
    //     }
    // }

    // for cityscapes dataset label
        void labelFilter()
    {
        for (int i = 0; i < N_SCAN; i++)
        {
            for (int j = 0; j < Horizon_SCAN; j++)
            {
                // // 跳过 NaN 点
                // if (rangeMatrix.at<float>(i, j) == -1 || rangeMatrix.at<float>(i + 1, j) == -1)
                //     continue;
                // if(labelMatrix.at<int>(i,j) == 0 || labelMatrix.at<int>(i,j) == 1){
                //     semObstacleMatrix.at<int>(i,j) = 0; // 可通行
                // } else if(labelMatrix.at<int>(i,j) != -2) { // label is not "flat"
                //     // ROS_INFO("label: %d",labelMatrix.at<int>(i,j));
                //     semObstacleMatrix.at<int>(i,j) = 1;   // 静态障碍物
                // }else if(labelMatrix.at<int>(i,j) >= 11){ // 假设10是动态障碍物
                //     // ROS_INFO("dynamic obstacle label: %d",labelMatrix.at<int>(i,j));
                //     semObstacleMatrix.at<int>(i,j) = 2;  // 动态障碍物
                // }
                // 跳过 NaN 点
                if (rangeMatrix.at<float>(i, j) == -1 || rangeMatrix.at<float>(i + 1, j) == -1)
                    continue;
                if(labelMatrix.at<int>(i,j) == 0){ // 可通行
                    semObstacleMatrix.at<int>(i,j) = 0; 
                } else if(labelMatrix.at<int>(i,j) == 2) { // 静态障碍物
                    // ROS_INFO("label: %d",labelMatrix.at<int>(i,j));
                    semObstacleMatrix.at<int>(i,j) = 1;   
                }else if(labelMatrix.at<int>(i,j) == 11){ // 动态障碍物
                    // ROS_INFO("dynamic obstacle label: %d",labelMatrix.at<int>(i,j));
                    semObstacleMatrix.at<int>(i,j) = 2; 
                }
            }
        }
    }
    /*
    // * original version of the positive filter
    void positiveCurbFilter()
    {
        int rangeCompareNeighborNum = 3;
        float diff[Horizon_SCAN - 1];

        for (int i = 0; i < scanNumCurbFilter; ++i){
            // calculate range difference
            for (int j = 0; j < Horizon_SCAN - 1; ++j)
                diff[j] = rangeMatrix.at<float>(i, j) - rangeMatrix.at<float>(i, j+1); // 同一个ring上相邻点的range difference

            for (int j = rangeCompareNeighborNum; j < Horizon_SCAN - rangeCompareNeighborNum; ++j){
                // Point that has been verified by other filters
                if (obstacleMatrix.at<int>(i, j) == 1) {
                    continue;
                }
                bool breakFlag = false;
                // point is too far away, skip comparison since it can be inaccurate
                if (rangeMatrix.at<float>(i, j) > sensorRangeLimit) {
                    continue;
                }
                // make sure all points have valid range info
                for (int k = -rangeCompareNeighborNum; k <= rangeCompareNeighborNum; ++k) {
                    if (rangeMatrix.at<float>(i, j+k) == -1){
                        breakFlag = true;
                        break;
                    }
                }
                if (breakFlag == true) { continue; }
                // range difference should be monotonically increasing or decresing
                for (int k = -rangeCompareNeighborNum; k < rangeCompareNeighborNum-1; ++k)
                    if (diff[j+k] * diff[j+k+1] <= 0){
                        breakFlag = true;
                        break;
                    }
                if (breakFlag == true) { continue; }
                // the range difference between the start and end point of neighbor points is smaller than a threashold, then continue
                if (fabs(rangeMatrix.at<float>(i, j-rangeCompareNeighborNum) - rangeMatrix.at<float>(i, j+rangeCompareNeighborNum)) /rangeMatrix.at<float>(i, j) < 0.01) {
                    continue;
                }
                obstacleMatrix.at<int>(i, j) = 1;
            }
        }
    }
    */

    // * second version of positive filter, by checking the normal vector of the neighbor points
    void positiveCurbFilter_v2()
    {
        int rangeNormCalculation = 6;
        for (int i = 0; i < scanNumCurbFilter; i++)
        {
            for (int j = rangeNormCalculation; j < Horizon_SCAN - rangeNormCalculation; j++)
            {
                // 跳过已判断的点
                if (geoObstacleMatrix.at<int>(i, j) == 1 || semObstacleMatrix.at<int>(i, j) == 2)
                {
                    continue;
                }
                if (rangeMatrix.at<float>(i, j) > sensorRangeLimit || rangeMatrix.at<float>(i, j) == -1)
                {
                    continue;
                }
                // int index = j  + i * Horizon_SCAN;
                vector<float> neighborArray;
                for (int k = -rangeNormCalculation; k <= rangeNormCalculation; k++)
                {
                    if (rangeMatrix.at<float>(i, j + k) != -1)
                    {
                        neighborArray.push_back(laserCloudMatrix[i][j + k].x);
                        neighborArray.push_back(laserCloudMatrix[i][j + k].y);
                        neighborArray.push_back(laserCloudMatrix[i][j + k].z);
                    }
                }
                if (i - 1 >= 0)
                {
                    for (int k = -rangeNormCalculation; k <= rangeNormCalculation; k++)
                    {
                        if (rangeMatrix.at<float>(i - 1, j + k) != -1)
                        {
                            neighborArray.push_back(laserCloudMatrix[i - 1][j + k].x);
                            neighborArray.push_back(laserCloudMatrix[i - 1][j + k].y);
                            neighborArray.push_back(laserCloudMatrix[i - 1][j + k].z);
                        }
                    }
                }
                if (i + 1 < scanNumCurbFilter)
                {
                    for (int k = -rangeNormCalculation; k <= rangeNormCalculation; k++)
                    {
                        if (rangeMatrix.at<float>(i + 1, j + k) != -1)
                        {
                            neighborArray.push_back(laserCloudMatrix[i + 1][j + k].x);
                            neighborArray.push_back(laserCloudMatrix[i + 1][j + k].y);
                            neighborArray.push_back(laserCloudMatrix[i + 1][j + k].z);
                        }
                    }
                }
                Eigen::MatrixXf matPoints = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(neighborArray.data(), neighborArray.size() / 3, 3);
                Eigen::MatrixXf centered = matPoints.rowwise() - matPoints.colwise().mean();
                Eigen::MatrixXf cov = (centered.adjoint() * centered); // 协方差矩阵
                Eigen::EigenSolver<Eigen::Matrix3f> es(cov);
                Eigen::Matrix3f D = es.pseudoEigenvalueMatrix();
                Eigen::Matrix3f V = es.pseudoEigenvectors();
                int rowIdx, colIdx;
                D.minCoeff(&rowIdx, &colIdx);
                Eigen::Vector3f normal = V.col(colIdx);
                normal /= normal.norm();
                // cout << "normal = " << normal.transpose() << endl;
                float slopeAngle = std::acos(std::fabs(normal(2))) / M_PI * 180;
                if (fabs(slopeAngle - 90) > 10.0)
                {
                    geoObstacleMatrix.at<int>(i, j) = 1;
                }
            }
        }
    }

    /*
    // * first version of customized positive curb filter, by checking the monotonicity of the neighbor points
    void positiveCurbFilter()
    {
        int rangeCompareNeighborNum = 3;
        float diff[Horizon_SCAN - 1];

        for (int i = 0; i < scanNumCurbFilter; ++i){

            for (int j = rangeCompareNeighborNum; j < Horizon_SCAN - rangeCompareNeighborNum; ++j){
                // Point that has been verified by other filters
                if (obstacleMatrix.at<int>(i, j) == 1) {
                    continue;
                }
                bool breakFlag = false;
                // point is too far away, skip comparison since it can be inaccurate
                if (rangeMatrix.at<float>(i, j) > sensorRangeLimit || rangeMatrix.at<float>(i,j) == -1) {
                    continue;
                }
                vector<float> neighborArray; // valid neighboring points (in order)
                float minZ = FLT_MAX;
                float maxZ = FLT_MIN;
                for (int k = -rangeCompareNeighborNum; k <= rangeCompareNeighborNum; k++) {
                    if (rangeMatrix.at<float>(i,j+k) != -1) {
                        neighborArray.push_back(rangeMatrix.at<float>(i,j+k));
                        minZ = min(laserCloudMatrix[i][j].z, minZ);
                        maxZ = max(laserCloudMatrix[i][j].z, maxZ);
                    }
                }

                float minR = *std::min_element(neighborArray.begin(),neighborArray.end());
                float maxR = *std::max_element(neighborArray.begin(),neighborArray.end());
                float max_diff = maxR - minR;
                // check monotonic
                if (!isMonotonic(neighborArray)) continue;
                if (max_diff / rangeMatrix.at<float>(i, j) < 0.02 ) {
                    continue;
                }
                obstacleMatrix.at<int>(i, j) = 1;
            }
        }
    }

    bool isMonotonic(vector<float>& num) {
        return checkMonotonic(num, true) || checkMonotonic(num, false);
    }
    bool checkMonotonic(vector<float>& num, bool flag) {
        for (int i = 0; i < num.size() - 1; i++) {
            if (flag) {
                if (num[i] > num[i+1]) return false;
            }
            else {
                if (num[i] < num[i+1]) return false;
            }
        }
        return true;
    }
    */

    // * filter for negative curb structures
    void negativeCurbFilter()
    {
        int rangeCompareNeighborNum = 3;

        for (int i = 0; i < scanNumCurbFilter; ++i)
        {
            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                // Point that has been verified by other filters
                if (geoObstacleMatrix.at<int>(i, j) == 1 || semObstacleMatrix.at<int>(i, j) == 2)
                    continue;
                // point without range value cannot be verified
                if (rangeMatrix.at<float>(i, j) == -1)
                    continue;
                // point is too far away, skip comparison since it can be inaccurate
                if (rangeMatrix.at<float>(i, j) > sensorRangeLimit)
                    continue;
                // check neighbors
                for (int m = -rangeCompareNeighborNum; m <= rangeCompareNeighborNum; ++m)
                {
                    int k = j + m;
                    if (k < 0 || k >= Horizon_SCAN)
                        continue;
                    if (rangeMatrix.at<float>(i, k) == -1)
                        continue;
                    // height diff greater than threshold, might be a negative curb

                    // if里的第一个条件要不要abs?
                    if (fabs(laserCloudMatrix[i][j].z - laserCloudMatrix[i][k].z) > 0.1 && pointDistance(laserCloudMatrix[i][j], laserCloudMatrix[i][k]) <= 1.0)
                    {
                        geoObstacleMatrix.at<int>(i, j) = 1;
                        break;
                    }
                }
            }
        }
    }

    // * filter for slope structures
    void slopeFilter()
    {

        for (int i = 0; i < scanNumSlopeFilter; ++i)
        {
            // 为什么slope也要限制scanNum
            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                // Point that has been verified by other filters
                if (geoObstacleMatrix.at<int>(i, j) == 1 || semObstacleMatrix.at<int>(i, j) == 2)
                    continue;
                // point without range value cannot be verified
                if (rangeMatrix.at<float>(i, j) == -1 || rangeMatrix.at<float>(i + 1, j) == -1)
                    continue;
                // point is too far away, skip comparison since it can be inaccurate
                if (rangeMatrix.at<float>(i, j) > sensorRangeLimit)
                    continue;
                // Two range filters here:
                // 1. if a point's range is larger than scanNumSlopeFilter th ring point's range
                // 2. if a point's range is larger than the upper point's range
                // then this point is very likely on obstacle. i.e. a point under the car or on a pole
                // if (  (rangeMatrix.at<float>(scanNumSlopeFilter, j) != -1 && rangeMatrix.at<float>(i, j) > rangeMatrix.at<float>(scanNumSlopeFilter, j))
                //     || (rangeMatrix.at<float>(i, j) > rangeMatrix.at<float>(i+1, j)) ){
                //     obstacleMatrix.at<int>(i, j) = 1;
                //     continue;
                // }
                // Calculate slope angle
                float diffX = laserCloudMatrix[i + 1][j].x - laserCloudMatrix[i][j].x; // 前后rings同一径向上比较
                float diffY = laserCloudMatrix[i + 1][j].y - laserCloudMatrix[i][j].y;
                float diffZ = laserCloudMatrix[i + 1][j].z - laserCloudMatrix[i][j].z;
                float angle = atan2(diffZ, sqrt(diffX * diffX + diffY * diffY)) * 180 / M_PI;
                // Slope angle is larger than threashold, mark as obstacle point
                if (angle < -filterAngleLimit || angle > filterAngleLimit)
                {
                    geoObstacleMatrix.at<int>(i, j) = 1;
                    continue;
                }
            }
        }
    }

    // * filter traversable regions using intensity, only tested on HALL7's sidewalk
    void intensityFilter()
    { 
        for (int i = 0; i < N_SCAN; i++)
        {
            for (int j = 0; j < Horizon_SCAN; j++)
            {
                // Point that has been verified by other filters
                if (geoObstacleMatrix.at<int>(i, j) == 1)
                    continue;
                if (rangeMatrix.at<float>(i, j) == -1 || rangeMatrix.at<float>(i + 1, j) == -1)
                    continue;
                // point is too far away, skip comparison since it can be inaccurate
                if (rangeMatrix.at<float>(i, j) > sensorRangeLimit)
                    continue;
                if (intensityMatrix.at<float>(i, j) > intensityLimit)
                {
                    geoObstacleMatrix.at<int>(i, j) = 1;
                    continue;
                }
            }
        }
    }

    // * labelled obstacle & free regions according to filter results
    void extractFilteredCloud()
    {
        for (int i = 0; i < scanNumMax; ++i)
        { // scanNumMax = std::max(scanNumCurbFilter, scanNumSlopeFilter) = max(8,5)
            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                // invalid points and points too far are skipped
                if (rangeMatrix.at<float>(i, j) > sensorRangeLimit ||
                    rangeMatrix.at<float>(i, j) == -1)
                    continue;
                // update point intensity (occupancy) into
                PointType p = laserCloudMatrix[i][j];
                // 根据语义障碍物和几何障碍物矩阵，进行映射并赋给intensity
                if (semObstacleMatrix.at<int>(i, j) == 2)
                {
                    p.intensity = 50; // 表示语义信息判断为动态障碍物
                }
                else if (geoObstacleMatrix.at<int>(i, j) == 0)
                {
                    if (semObstacleMatrix.at<int>(i, j) == -1)
                    {
                        p.intensity = 0; // 表示不含语义信息，几何信息判断为非障碍物
                    }
                    else if (semObstacleMatrix.at<int>(i, j) == 0)
                    {
                        p.intensity = 10; // 表示语义信息判断为非障碍物，几何信息判断为非障碍物
                    }
                    else if (semObstacleMatrix.at<int>(i, j) == 1)
                    {
                        p.intensity = 30; // 表示语义信息判断为静态障碍物，几何信息判断为非障碍物
                    }
                }
                else if (geoObstacleMatrix.at<int>(i, j) == 1)
                {
                    if (semObstacleMatrix.at<int>(i, j) == -1)
                    {
                        p.intensity = 60; // 表示不含语义信息，几何信息判断为障碍物
                    }
                    else if (semObstacleMatrix.at<int>(i, j) == 0)
                    {
                        p.intensity = 80; // 表示语义判断为非障碍物，几何判断为障碍物
                    }
                    else if (semObstacleMatrix.at<int>(i, j) == 1)
                    {
                        p.intensity = 100; // 表示语义判断为静态障碍物，几何判断为障碍物
                    }
                }

                // save updated points
                laserCloudOut->push_back(p);
                // extract obstacle points and convert them to laser scan
                if (p.intensity == 100)
                    laserCloudObstacles->push_back(p);
            }
        }

        // Publish laserCloudOut for visualization (before downsample and BGK prediction)
        if (pubCloudVisualHiRes.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 laserCloudTemp;
            pcl::toROSMsg(*laserCloudOut, laserCloudTemp);
            laserCloudTemp.header.stamp = ros::Time::now();
            // laserCloudTemp.header.stamp = pcMsgTimeStamp;
            laserCloudTemp.header.frame_id = "map";
            pubCloudVisualHiRes.publish(laserCloudTemp);
        }
    }

    //* downsample raw cloud to grid
    void downsampleCloud()
    {
        float roundedX = float(int(robotPoint.x * 10.0f)) / 10.0f;
        float roundedY = float(int(robotPoint.y * 10.0f)) / 10.0f;
        float roundedZ = float(int(robotPoint.z * 10.0f)) / 10.0f;
        // height map origin   以机器人位置(x-10,y-10) 作为起点
        localMapOrigin.x = roundedX - sensorRangeLimit;
        localMapOrigin.y = roundedY - sensorRangeLimit;

        unordered_map<int, vector<float>> heightmap; // 存储grid map上每个点对应的高度 laserCloudOut->points[i].z
        int helper = filterHeightMapArrayLength + 1;

        // convert from point cloud to height map
        int cloudSize = laserCloudOut->points.size();
        for (int i = 0; i < cloudSize; ++i)
        {

            int idx = (laserCloudOut->points[i].x - localMapOrigin.x) / mapResolution;
            int idy = (laserCloudOut->points[i].y - localMapOrigin.y) / mapResolution;

            int linear_idx = idx * helper + idy; // easy from linear_idx to idx / idy
            // points out of boundry
            if (idx < 0 || idy < 0 || idx >= filterHeightMapArrayLength || idy >= filterHeightMapArrayLength)
                continue;
            // obstacle point (decided by curb or slope filter)
            if (semObstFlag[idx][idy] != 2)
            { // 如果不是动态标签，则持续更新
                if (laserCloudOut->points[i].intensity == 50)
                {
                    semObstFlag[idx][idy] = 2;
                }
                else if (geoObstFlag[idx][idy] == true)
                {
                    if (semObstFlag[idx][idy] != 1)
                    {
                        if (laserCloudOut->points[i].intensity == 100 || laserCloudOut->points[i].intensity == 30)
                    {
                        semObstFlag[idx][idy] = 1;
                    }
                    else if (laserCloudOut->points[i].intensity == 80 || laserCloudOut->points[i].intensity == 10)
                    {
                        semObstFlag[idx][idy] = 0;
                    }
                    }

                }
                else{
                    if (semObstFlag[idx][idy] == 1)
                    {
                        if (laserCloudOut->points[i].intensity >= 60)
                        { // 60, 80, 100
                            geoObstFlag[idx][idy] = true;
                        }
                    }
                    else if (laserCloudOut->points[i].intensity == 10)
                    {
                        semObstFlag[idx][idy] = 0;
                    }
                    else if (laserCloudOut->points[i].intensity == 30)
                    {
                        semObstFlag[idx][idy] = 1;
                    }
                    else if (laserCloudOut->points[i].intensity == 60)
                    {
                        geoObstFlag[idx][idy] = true;
                    }
                    else if (laserCloudOut->points[i].intensity == 80)
                    {
                        semObstFlag[idx][idy] = 0;
                        geoObstFlag[idx][idy] = true;
                    }
                    else if (laserCloudOut->points[i].intensity == 100)
                    {
                        semObstFlag[idx][idy] = 1;
                        geoObstFlag[idx][idy] = true;
                    }
                }
            }

            heightmap[linear_idx].push_back(laserCloudOut->points[i].z);
            if (initFlag[idx][idy] == false)
            {
                minHeight[idx][idy] = laserCloudOut->points[i].z;
                maxHeight[idx][idy] = laserCloudOut->points[i].z;
                initFlag[idx][idy] = true;
            }
            else
            {
                minHeight[idx][idy] = std::min(minHeight[idx][idy], laserCloudOut->points[i].z);
                maxHeight[idx][idy] = std::max(maxHeight[idx][idy], laserCloudOut->points[i].z);
            }
        }

        float vehicleHeight = 1.0;



        // gjj新增：将机器人周围的栅格设为可通行
        int robot_idx = sensorRangeLimit/mapResolution - robotArrayLength/2;
        int robot_idy = sensorRangeLimit/mapResolution - robotArrayLength/2;
        for(int i = 0;i < robotArrayLength;i++){
            for(int j = 0;j < robotArrayLength;j++){
                int idx = robot_idx+i;
                int idy = robot_idy+j;
                semObstFlag[idx][idy] = 0;
                geoObstFlag[idx][idy] = false;

                if(initFlag[idx][idy] == false){
                    minHeight[idx][idy] = roundedZ-0.22;
                    maxHeight[idx][idy] = roundedZ-0.22;
                    initFlag[idx][idy] = true;
                }
            }
        }

        /*
        // process hanging over structure
        for (auto it = heightmap.begin(); it != heightmap.end(); it++) {
            int idx = it->first / helper;
            int idy = it->first % helper;
            auto& heightPoint = it->second;
            cout << "DEBUGGING EXAMPLE: height point size at (" << idx << "," << idy << ") is " << heightPoint.size() << endl;
            sort(heightPoint.begin(),heightPoint.end());
            float lowestHeight = 0;
            for (int i = 0; i < heightPoint.size() - 1; i++) {
                if (i == 0) lowestHeight = heightPoint[i];
                else {
                    if (heightPoint[i+1] - heightPoint[i] > vehicleHeight && heightPoint[i] - lowestHeight < traversableHeight) {
                        obstFlag[idx][idy] = true; // 应该是标记成clear的，此处仅供测试用
                    }
                }
            }
        }
        */
        // intermediate cloud (adding process for hanging over structure)
        pcl::PointCloud<PointType>::Ptr laserCloudTemp(new pcl::PointCloud<PointType>());
        // convert from height map to point cloud
        for (int i = 0; i < filterHeightMapArrayLength; ++i)
        {
            for (int j = 0; j < filterHeightMapArrayLength; ++j)
            {
                // no point at this grid
                if (initFlag[i][j] == false)
                    continue;
                // convert grid to point
                PointType thisPoint;
                thisPoint.x = localMapOrigin.x + i * mapResolution + mapResolution / 2.0;
                thisPoint.y = localMapOrigin.y + j * mapResolution + mapResolution / 2.0;
                thisPoint.z = maxHeight[i][j]; // downsample时只记录最大

                if (geoObstFlag[i][j] == true || maxHeight[i][j] - minHeight[i][j] > filterHeightLimit)
                    if (maxHeight[i][j] - minHeight[i][j] > filterHeightLimit)
                    { // 对于之前没有被识别为障碍物的点，如果映射到同一网格的高度差太大也会再次被设为障碍物点
                        geoObstFlag[i][j] = true;
                        /*
                        int linear_idx = i * helper + j;
                        if (heightmap.find(linear_idx) != heightmap.end()) {
                            // multiply height value exist
                            auto& heightVec = heightmap[linear_idx];
                            // cout << "DEBUGGING EXAMPLE: height point size at (" << i << "," << j << ") is " << heightVec.size() << endl;
                            sort(heightVec.begin(),heightVec.end());
                            float lowestHeight = 0;
                            for (int k = 0; k < heightVec.size() - 1; k++) {
                                if (k == 0) lowestHeight =intensityalse; // 若存在hanging over，则重新置为false
                                    }
                                }
                            }
                        }
                        */
                    }
                // if (geoObstFlag[i][j] == true){
                //     thisPoint.intensity = 100; // obstacle
                //     laserCloudTemp->push_back(thisPoint);
                //     // laserCloudObstacles->push_back(thisPoint);
                // }else if(semObstFlag[i][j] == true){
                //     thisPoint.intensity = 50; // dynamic obstacle 设为50
                //     laserCloudTemp->push_back(thisPoint);
                // }
                // else{
                //     thisPoint.intensity = 0; // free
                //     laserCloudTemp->push_back(thisPoint);
                //     //////////////////////////////////////
                //     //////// 如果分成两个point cloud发送呢？
                //     //////////////////////////////////////
                // }
                // 根据语义障碍物和几何障碍物矩阵，进行映射并赋给intensity
                if (semObstFlag[i][j] == 2)
                {
                    thisPoint.intensity = 50; // 表示语义信息判断为动态障碍物
                }
                else if (geoObstFlag[i][j] == false)
                {
                    if (semObstFlag[i][j] == -1)
                    {
                        thisPoint.intensity = 0; // 表示不含语义信息，几何信息判断为非障碍物
                    }
                    else if (semObstFlag[i][j] == 0)
                    {
                        thisPoint.intensity = 10; // 表示语义信息判断为非障碍物，几何信息判断为非障碍物
                    }
                    else if (semObstFlag[i][j] == 1)
                    {
                        thisPoint.intensity = 30; // 表示语义信息判断为静态障碍物，几何信息判断为非障碍物
                    }
                }
                else                                //geoObstFlag[i][j] 为 true
                {
                    if (semObstFlag[i][j] == -1)
                    {
                        thisPoint.intensity = 60; // 表示不含语义信息，几何信息判断为障碍物
                    }
                    else if (semObstFlag[i][j] == 0)
                    {
                        thisPoint.intensity = 80; // 表示语义判断为非障碍物，几何判断为障碍物
                    }
                    else if (semObstFlag[i][j] == 1)
                    {
                        thisPoint.intensity = 100; // 表示语义判断为静态障碍物，几何判断为障碍物
                    }
                }
                laserCloudTemp->push_back(thisPoint);
                 
                // if (thisPoint.intensity)
                // {
                //     ROS_INFO("intensity: %f", thisPoint.intensity);
                // }
            }
        }

        *laserCloudOut = *laserCloudTemp; // 只有前8线的

        // Publish laserCloudOut for visualization (after downsample but beforeBGK prediction)
        if (pubCloudVisualLowRes.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 laserCloudTemp;
            pcl::toROSMsg(*laserCloudOut, laserCloudTemp);
            laserCloudTemp.header.stamp = ros::Time::now();
            // laserCloudTemp.header.stamp = pcMsgTimeStamp;
            laserCloudTemp.header.frame_id = ns+"/map";
            pubCloudVisualLowRes.publish(laserCloudTemp);
        }
    }

    //*  BGK inference to produce more point info
    void predictCloudBGK()
    {

        if (predictionEnableFlag == false)
            return;

        int kernelGridLength = int(predictionKernalSize / mapResolution);
        int cloudSize = laserCloudOut->points.size(); // 记录已有的点云个数

        // 推断静态障碍物
        for (int i = 0; i < filterHeightMapArrayLength; ++i)
        {
            for (int j = 0; j < filterHeightMapArrayLength; ++j)
            {
                // skip observed point
                if (initFlag[i][j] == true)
                    continue;
                PointType testPoint;
                testPoint.x = localMapOrigin.x + i * mapResolution + mapResolution / 2.0;
                testPoint.y = localMapOrigin.y + j * mapResolution + mapResolution / 2.0;
                testPoint.z = robotPoint.z; // this value is not used except for computing distance with robotPoint
                // skip grids too far
                if (pointDistance(testPoint, robotPoint) > sensorRangeLimit)
                    continue;
                // Training data
                vector<float> xTrainVec;     // training data x and y coordinates
                vector<float> yTrainVecElev; // training data elevation
                vector<float> yTrainVecOccu; // training data occupancy
                // Fill trainig data (vector)
                for (int m = -kernelGridLength; m <= kernelGridLength; ++m)
                {
                    for (int n = -kernelGridLength; n <= kernelGridLength; ++n)
                    {
                        // skip grids too far
                        if (std::sqrt(float(m * m + n * n)) * mapResolution > predictionKernalSize)
                            continue;
                        int idx = i + m;
                        int idy = j + n;
                        // index out of boundry
                        if (idx < 0 || idy < 0 || idx >= filterHeightMapArrayLength || idy >= filterHeightMapArrayLength)
                            continue;
                        // save only observed grid in this scan
                        if (initFlag[idx][idy] == true)
                        {
                            xTrainVec.push_back(localMapOrigin.x + idx * mapResolution + mapResolution / 2.0);
                            xTrainVec.push_back(localMapOrigin.y + idy * mapResolution + mapResolution / 2.0);
                            yTrainVecElev.push_back(maxHeight[idx][idy]);
                            yTrainVecOccu.push_back((geoObstFlag[idx][idy]== true || semObstFlag[idx][idy]==1)  ? 1 : 0);
                        }
                    }
                }
                // no training data available, continue
                if (xTrainVec.size() == 0)
                    continue;
                // convert from vector to eigen
                Eigen::MatrixXf xTrain = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(xTrainVec.data(), xTrainVec.size() / 2, 2);
                Eigen::MatrixXf yTrainElev = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(yTrainVecElev.data(), yTrainVecElev.size(), 1);
                Eigen::MatrixXf yTrainOccu = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(yTrainVecOccu.data(), yTrainVecOccu.size(), 1);
                // Test data (current grid)
                vector<float> xTestVec;
                xTestVec.push_back(testPoint.x);
                xTestVec.push_back(testPoint.y);
                Eigen::MatrixXf xTest = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(xTestVec.data(), xTestVec.size() / 2, 2);
                // Predict
                Eigen::MatrixXf Ks;           // covariance matrix
                covSparse(xTest, xTrain, Ks); // sparse kernel

                Eigen::MatrixXf ybarElev = (Ks * yTrainElev).array();
                Eigen::MatrixXf ybarOccu = (Ks * yTrainOccu).array();
                Eigen::MatrixXf kbar = Ks.rowwise().sum().array();

                // Update Elevation with Prediction
                if (std::isnan(ybarElev(0, 0)) || std::isnan(ybarOccu(0, 0)) || std::isnan(kbar(0, 0)))
                    continue;

                if (kbar(0, 0) == 0)
                    continue;

                float elevation = ybarElev(0, 0) / kbar(0, 0);
                float occupancy = ybarOccu(0, 0) / kbar(0, 0);

                PointType p;
                p.x = xTestVec[0];
                p.y = xTestVec[1];
                p.z = elevation;
                p.intensity = (occupancy > 0.7) ? 60 : 0;  // // 60 表示不含语义信息，几何判断为障碍物； 表示不含语义信息，几何信息判断为非障碍物

                laserCloudOut->push_back(p);
            }
        }

        // 推断动态障碍物
        // for (int i = 0; i < filterHeightMapArrayLength; ++i){
        //     for (int j = 0; j < filterHeightMapArrayLength; ++j){
        //         // skip observed point
        //         if (initFlag[i][j] == true)
        //             continue;
        //         PointType testPointDyn;
        //         testPointDyn.x = localMapOrigin.x + i * mapResolution + mapResolution / 2.0;
        //         testPointDyn.y = localMapOrigin.y + j * mapResolution + mapResolution / 2.0;
        //         testPointDyn.z = robotPoint.z; // this value is not used except for computing distance with robotPoint
        //         // skip grids too far
        //         if (pointDistance(testPointDyn, robotPoint) > sensorRangeLimit)
        //             continue;
        //         // Training data
        //         vector<float> xTrainVecDyn; // training data x and y coordinates
        //         vector<float> yTrainVecElevDyn; // training data elevation
        //         vector<float> yTrainVecOccuDyn; // training data occupancy
        //         // Fill trainig data (vector)
        //         for (int m = -kernelGridLength; m <= kernelGridLength; ++m){
        //             for (int n = -kernelGridLength; n <= kernelGridLength; ++n){
        //                 // skip grids too far
        //                 if (std::sqrt(float(m*m + n*n)) * mapResolution > predictionKernalSize)
        //                     continue;
        //                 int idx = i + m;
        //                 int idy = j + n;
        //                 // index out of boundry
        //                 if (idx < 0 || idy < 0 || idx >= filterHeightMapArrayLength || idy >= filterHeightMapArrayLength)
        //                     continue;
        //                 // save only observed grid in this scan
        //                 if (initFlag[idx][idy] == true){
        //                     xTrainVecDyn.push_back(localMapOrigin.x + idx * mapResolution + mapResolution / 2.0);
        //                     xTrainVecDyn.push_back(localMapOrigin.y + idy * mapResolution + mapResolution / 2.0);
        //                     yTrainVecElevDyn.push_back(maxHeight[idx][idy]);
        //                     yTrainVecOccuDyn.push_back(dynObstFlag[idx][idy] == true ? 1:0);
        //                 }
        //             }
        //         }
        //         // no training data available, continue
        //         if (xTrainVecDyn.size() == 0)
        //             continue;
        //         // convert from vector to eigen
        //         Eigen::MatrixXf xTrainDyn = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(xTrainVecDyn.data(), xTrainVecDyn.size() / 2, 2);
        //         Eigen::MatrixXf yTrainElevDyn = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(yTrainVecElevDyn.data(), yTrainVecElevDyn.size(), 1);
        //         Eigen::MatrixXf yTrainOccuDyn = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(yTrainVecOccuDyn.data(), yTrainVecOccuDyn.size(), 1);
        //         // Test data (current grid)
        //         vector<float> xTestVecDyn;
        //         xTestVecDyn.push_back(testPointDyn.x);
        //         xTestVecDyn.push_back(testPointDyn.y);
        //         Eigen::MatrixXf xTestDyn = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(xTestVecDyn.data(), xTestVecDyn.size() / 2, 2);
        //         // Predict
        //         Eigen::MatrixXf KsDyn; // covariance matrix
        //         covSparse(xTestDyn, xTrainDyn, KsDyn); // sparse kernel

        //         Eigen::MatrixXf ybarElevDyn = (KsDyn * yTrainElevDyn).array();
        //         Eigen::MatrixXf ybarOccuDyn = (KsDyn * yTrainOccuDyn).array();
        //         Eigen::MatrixXf kbarDyn = KsDyn.rowwise().sum().array();

        //         // Update Elevation with Prediction
        //         if (std::isnan(ybarElevDyn(0,0)) || std::isnan(ybarOccuDyn(0,0)) || std::isnan(kbarDyn(0,0)))
        //             continue;

        //         if (kbarDyn(0,0) == 0)
        //             continue;

        //         float elevationDyn = ybarElevDyn(0,0) / kbarDyn(0,0);
        //         float occupancyDyn = ybarOccuDyn(0,0) / kbarDyn(0,0);

        //         // PointType pDyn;
        //         // pDyn.x = xTestVecDyn[0];
        //         // pDyn.y = xTestVecDyn[1];
        //         // pDyn.z = elevationDyn;
        //         // pDyn.intensity = (occupancyDyn > 0.7) ? 50 : 0;
        //         // laserCloudOut->push_back(pDyn);
        //         if(occupancyDyn > 0.7) laserCloudOut->points[cloudSize + i*filterHeightMapArrayLength+j].intensity = 50;
        //     }
        // }
    }

    void dist(const Eigen::MatrixXf &xStar, const Eigen::MatrixXf &xTrain, Eigen::MatrixXf &d) const
    {
        d = Eigen::MatrixXf::Zero(xStar.rows(), xTrain.rows());
        for (int i = 0; i < xStar.rows(); ++i)
        {
            d.row(i) = (xTrain.rowwise() - xStar.row(i)).rowwise().norm();
        }
    }

    void covSparse(const Eigen::MatrixXf &xStar, const Eigen::MatrixXf &xTrain, Eigen::MatrixXf &Kxz) const
    {
        dist(xStar / (predictionKernalSize + 0.1), xTrain / (predictionKernalSize + 0.1), Kxz);
        Kxz = (((2.0f + (Kxz * 2.0f * 3.1415926f).array().cos()) * (1.0f - Kxz.array()) / 3.0f) +
               (Kxz * 2.0f * 3.1415926f).array().sin() / (2.0f * 3.1415926f))
                  .matrix() *
              1.0f;
        // Clean up for values with distance outside length scale, possible because Kxz <= 0 when dist >= predictionKernalSize
        for (int i = 0; i < Kxz.rows(); ++i)
            for (int j = 0; j < Kxz.cols(); ++j)
                if (Kxz(i, j) < 0)
                    Kxz(i, j) = 0;
    }

    void publishCloud()
    {
        sensor_msgs::PointCloud2 laserCloudTemp;
        pcl::toROSMsg(*laserCloudOut, laserCloudTemp);
        laserCloudTemp.header.stamp = ros::Time::now();
        // laserCloudTemp.header.stamp = pcMsgTimeStamp;
        laserCloudTemp.header.frame_id = ns+"/map";
        pubCloud.publish(laserCloudTemp);
    }

    void publishLaserScan()
    {

        updateLaserScan();

        laserScan.header.stamp = ros::Time::now();
        // laserScan.header.stamp = pcMsgTimeStamp;
        pubLaserScan.publish(laserScan);
        // initialize laser scan for new scan
        std::fill(laserScan.ranges.begin(), laserScan.ranges.end(), laserScan.range_max + 1.0);
    }

    void updateLaserScan()
    {

        try
        {
            listener.lookupTransform(ns+"/base_link", ns+"/map", ros::Time(0), transform);
        } // 从map到base_link的映射
        // try{listener.lookupTransform("base_link","map", pcMsgTimeStamp, transform);}

        catch (tf::TransformException ex)
        { /*ROS_ERROR("Transfrom Failure.");*/
            return;
        }

        laserCloudObstacles->header.frame_id = ns+"/map";
        laserCloudObstacles->header.stamp = 0;
        // transform obstacle cloud back to "base_link" frame
        pcl::PointCloud<PointType> laserCloudTemp;

        // 将map坐标系下的点云laserCloudObstacles （scanNumMax*Horizon_SCAN），映射到机器人坐标系下的点云laserCloudTemp
        pcl_ros::transformPointCloud(ns+"/base_link", *laserCloudObstacles, laserCloudTemp, listener); // frame_name, *pcl_in, pcl_out, *tf_listener

        // convert point to scan
        int cloudSize = laserCloudTemp.points.size();
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType *point = &laserCloudTemp.points[i];
            float x = point->x;
            float y = point->y;
            float range = std::sqrt(x * x + y * y);
            float angle = std::atan2(y, x);
            int index = (angle - laserScan.angle_min) / laserScan.angle_increment; // 根据角度，线性地计算出对应的index
            if (index >= 0 && index < laserScan.ranges.size())
                laserScan.ranges[index] = std::min(laserScan.ranges[index], range); // laserScan.ranges记录到障碍点的最短距离
        }
    }

    void pointcloud2laserscanInitialization()
    {

        laserScan.header.frame_id = ns+"/base_link"; // assume laser has the same frame as the robot

        laserScan.angle_min = -M_PI;
        laserScan.angle_max = M_PI;
        laserScan.angle_increment = 1.0f / 180 * M_PI;
        laserScan.time_increment = 0;

        laserScan.scan_time = 0.1;
        laserScan.range_min = 0.3;
        laserScan.range_max = 100;

        int range_size = std::ceil((laserScan.angle_max - laserScan.angle_min) / laserScan.angle_increment);
        laserScan.ranges.assign(range_size, laserScan.range_max + 1.0);
    }
};

int main(int argc, char **argv)
{

    ros::init(argc, argv, "traversability_mapping");

    TraversabilityFilter TFilter;

    ROS_INFO("\033[1;32m---->\033[0m Traversability Filter Started.");

    ros::spin();

    return 0;
}
