#include "utility.h"
#include "elevation_msgs/occupancyLocal.h"
class TraversabilityMapping{

private:

    // ROS Node Handle
    ros::NodeHandle nh;
    // Mutex Memory Lock
    std::mutex mtx;
    // Transform Listener
    tf::TransformListener listener;
    tf::StampedTransform transform;
    // Subscriber
    ros::Subscriber subFilteredGroundCloud;
    // Publisher
    ros::Publisher pubOccupancyMapLocal;
    ros::Publisher pubOccupancyMapLocalHeight;
    ros::Publisher pubGeoOccupancyMapLocal;
    ros::Publisher pubSemOccupancyMapLocal;
    ros::Publisher pubElevationCloud;
    // Point Cloud Pointer
    pcl::PointCloud<PointType>::Ptr laserCloud; // save input filtered laser cloud for mapping
    pcl::PointCloud<PointType>::Ptr laserCloudElevation; // a cloud for publishing elevation map
    // Occupancy Grid Map
    nav_msgs::OccupancyGrid geo_OccupancyMap2D; // local occupancy grid map for geometry info
    nav_msgs::OccupancyGrid sem_OccupancyMap2D; // local occupancy grid map for semantic info
    elevation_msgs::OccupancyElevation occupancyMap2DHeight; // customized message that includes occupancy map and elevation info

    int pubCount;
    
    // Map Arrays
    int mapArrayCount;
    int **mapArrayInd; // it saves the index of this submap in vector mapArray
    vector<childMap_t*> mapArray;

    // Local Map Extraction
    PointType robotPoint;
    PointType localMapOriginPoint;
    grid_t localMapOriginGrid;

    // Global Variables for Traversability Calculation
    cv::Mat matCov, matEig, matVec;

    // Lists for New Scan
    vector<mapCell_t*> observingList1; // thread 1: save new observed cells
    vector<mapCell_t*> observingList2; // thread 2: calculate traversability of new observed cells
    
    ros::Time initialTime_;
    string ns;  // namespace param
    
public:
    TraversabilityMapping():
        nh("~"),
        pubCount(1),
        mapArrayCount(0){
        if(!nh.getParam("namespace", ns)){
            ns = "";
        }
        cout<<"namespace is: "<<ns<<endl;
        // subscribe to traversability filter
        subFilteredGroundCloud = nh.subscribe<sensor_msgs::PointCloud2>(ns+"/filtered_pointcloud", 5, &TraversabilityMapping::cloudHandler, this);
        // subscribe directly to raw pointcloud (for elevation mapping test)
        // subFilteredGroundCloud = nh.subscribe<sensor_msgs::PointCloud2>("/filtered_pointcloud_visual_high_res", 5, &TraversabilityMapping::cloudHandler, this);
        // publish local occupancy and elevation grid map
        pubOccupancyMapLocal = nh.advertise<nav_msgs::OccupancyGrid> (ns+"/occupancy_map_local", 5);
        pubOccupancyMapLocalHeight = nh.advertise<elevation_msgs::OccupancyElevation> (ns+"/occupancy_map_local_height", 5);
        // publish elevation map for visualization
        pubElevationCloud = nh.advertise<sensor_msgs::PointCloud2> (ns+"/elevation_pointcloud", 5);
        initialTime_ = ros::Time::now();    

        allocateMemory(); 
    }

    ~TraversabilityMapping(){}
    
    void allocateMemory(){
        // allocate memory for point cloud
        laserCloud.reset(new pcl::PointCloud<PointType>());
        laserCloudElevation.reset(new pcl::PointCloud<PointType>());
        
        // initialize array for cmap
        mapArrayInd = new int*[mapArrayLength];
        for (int i = 0; i < mapArrayLength; ++i)
            mapArrayInd[i] = new int[mapArrayLength];

        for (int i = 0; i < mapArrayLength; ++i)
            for (int j = 0; j < mapArrayLength; ++j)
                mapArrayInd[i][j] = -1;


        // Matrix Initialization
        matCov = cv::Mat (3, 3, CV_32F, cv::Scalar::all(0));
        matEig = cv::Mat (1, 3, CV_32F, cv::Scalar::all(0));
        matVec = cv::Mat (3, 3, CV_32F, cv::Scalar::all(0));

        initializeLocalOccupancyMap();
    }


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////// Register Cloud /////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){
        // Lock thread
        std::lock_guard<std::mutex> lock(mtx);
        // Get Robot Position
        if (getRobotPosition() == false)   // 得到机器人map坐标系下的xyz坐标
            return;
        // Convert Point Cloud
        pcl::fromROSMsg(*laserCloudMsg, *laserCloud);
        ros::Time timestamp = laserCloudMsg->header.stamp;
        const float scanTimeSinceInitialization = (timestamp - initialTime_).toSec();
        // Register New Scan
        updateElevationMap(scanTimeSinceInitialization);
        // publish local occupancy grid map
        publishMap();
    }

    // * update elevation map
    void updateElevationMap(const float timestamp){
        int cloudSize = laserCloud->points.size();
        for (int i = 0; i < cloudSize; ++i){
            laserCloud->points[i].z -= 0.2; // for visualization
            updateElevationMap(&laserCloud->points[i], timestamp);
        }
    }

    // * update elevation map point by point
    void updateElevationMap(PointType *point, const float timestamp){

        // Find point index in global map
        grid_t thisGrid;
        if (findPointGridInMap(&thisGrid, point) == false) return;
        // Get current cell pointer
        mapCell_t *thisCell = grid2Cell(&thisGrid);
        // update elevation
        updateCellElevation(thisCell, point, timestamp);
        // update occupancy
        updateCellOccupancy(thisCell, point); // 待改进：occupancy信息也需要考虑方差（或者距离），太远的会有很多误检（大部分远的curb都会）
        // update observation time
        updateCellObservationTime(thisCell);
    }

    // * update observation time of cells in the elevation map
    void updateCellObservationTime(mapCell_t *thisCell){
        ++thisCell->observeTimes;
        // 在当前输入中有对应点的cell，且被观测到的时间超过阈值，就会被加到这个list（然后被丢去计算traversability）
        if (thisCell->observeTimes >= traversabilityObserveTimeTh)
            observingList1.push_back(thisCell);
    }

    // * update occupancy probability of cells in the elevation map
    void updateCellOccupancy(mapCell_t *thisCell, PointType *point)
    {
        float geo_p;  // Probability of being occupied knowing current measurement.
        float sem_p;
        bool sem = true;     // 判断是否含有语义信息
        
        // 特殊情况：动态点不更新log odds，occupancy设为100
        if(point->intensity == 50){ 
            thisCell->updateOccupancy(100);
            return;
        }
        float occupancy;

        // Update log_odds
        if (point->intensity == 100){  // 表示语义判断为静态障碍物，几何判断为障碍物
            geo_p = p_occupied_when_laser;
            sem_p = p_occupied_when_semantic;  
        }else if(point->intensity == 80){   // 表示语义判断为非障碍物，几何判断为障碍物
            geo_p = p_occupied_when_laser;
            sem_p = p_occupied_when_no_semantic;
        }else if(point->intensity == 60){   // 表示不含语义信息，几何判断为障碍物
            geo_p = p_occupied_when_laser;
            sem_p = 0.5;
            sem = false;
        }else if(point->intensity == 30){   // 表示语义信息判断为静态障碍物，几何信息判断为非障碍物
            geo_p = p_occupied_when_no_laser;
            sem_p = p_occupied_when_semantic;
        }else if(point->intensity == 10){   // 表示语义信息判断为非障碍物，几何信息判断为非障碍物
            geo_p = p_occupied_when_no_laser;
            sem_p = p_occupied_when_no_semantic;
        }else{                              // 表示不含语义信息，几何信息判断为非障碍物
            geo_p = p_occupied_when_no_laser;
            sem_p = 0.5;
            sem = false;
        }
        if(sem) thisCell->sem_obs_times += 1;  // 增加被语义观测到的次数

        // 更新几何和语义log odd
        thisCell->geo_log_odds += std::log(geo_p / (1 - geo_p)); // 更新 log odds
        thisCell->sem_log_odds += std::log(sem_p / (1 - sem_p)); // 更新 log odds
        // log odds取值是(-inf, inf)，这里手动限制为 (-100,100)
        if (thisCell->geo_log_odds < -large_log_odds) 
            thisCell->geo_log_odds = -large_log_odds;
        else if (thisCell->geo_log_odds > large_log_odds)
            thisCell->geo_log_odds = large_log_odds;

        if (thisCell->sem_log_odds < -large_log_odds) 
            thisCell->sem_log_odds = -large_log_odds;
        else if (thisCell->sem_log_odds > large_log_odds)
            thisCell->sem_log_odds = large_log_odds;
        
        // Update occupancy
        float geo_occupancy;
        float sem_occupancy;
        if (thisCell->geo_log_odds < -max_log_odds_for_belief){ // 小于-20设为非占据
            geo_occupancy = 0;
        } 
        else if (thisCell->geo_log_odds > max_log_odds_for_belief){ // 大于20设为占据
            geo_occupancy = 100;
        }
        else{
            geo_occupancy = lround((1 - 1 / (1 + std::exp(thisCell->geo_log_odds))) * 100);
        }

        if(thisCell->sem_obs_times > semObserveTime){
            if (thisCell->sem_log_odds < -max_log_odds_for_belief){  // 小于-20设为非占据
            sem_occupancy = 0;
            }
            else if (thisCell->sem_log_odds > max_log_odds_for_belief){ // 大于20设为占据
                sem_occupancy = 100;
            }
            else{
                sem_occupancy = lround((1 - 1 / (1 + std::exp(thisCell->sem_log_odds))) * 100);
            }
            // update cell
            occupancy = sem_occupancy*0.6 + geo_occupancy*0.4;
            if(occupancy>60) occupancy = 100;
        } else{
            occupancy = geo_occupancy;
            if(occupancy>70) occupancy = 100;
        } 
        // occupancy = geo_occupancy;
        // if(occupancy>70) occupancy = 100;
        thisCell->updateOccupancy(occupancy);
        // ROS_INFO("geo %f sem %f",geo_occupancy,sem_occupancy);
        // ROS_INFO("occupancy %f geo %f sem %f",occupancy,geo_occupancy,sem_occupancy);
    }

    // * update elevation information of cells in the elevation map
    void updateCellElevation(mapCell_t *thisCell, PointType *point, const float timestamp){
        // Kalman Filter: update cell elevation using Kalman filter
        // https://www.cs.cornell.edu/courses/cs4758/2012sp/materials/MI63slides.pdf

        // cell is observed for the first time, no need to use Kalman filter
        if (thisCell->elevation == -FLT_MAX){
            thisCell->elevation = point->z;
            thisCell->elevationVar = pointDistance(robotPoint, *point);
            return;
        }

        // previous kalman filter
        // Predict:
        float x_pred = thisCell->elevation; // x = F * x + B * u
        float P_pred = thisCell->elevationVar + 0.01; // P = F*P*F + Q
        // Update:
        float R_factor = (thisCell->observeTimes > 20) ? 10 : 1;
        float R = pointDistance(robotPoint, *point) * R_factor; // measurement noise: R, scale it with dist and observed times
        float K = P_pred / (P_pred + R);// Gain: K  = P * H^T * (HPH + R)^-1
        float y = point->z; // measurement: y
        float x_final = x_pred + K * (y - x_pred); // x_final = x_pred + K * (y - H * x_pred)
        float P_final = (1 - K) * P_pred; // P_final = (I - K * H) * P_pred
        // Update cell
        thisCell->updateElevation(x_final, P_final);
    }

    mapCell_t* grid2Cell(grid_t *thisGrid){
        return mapArray[mapArrayInd[thisGrid->cubeX][thisGrid->cubeY]]->cellArray[thisGrid->gridX][thisGrid->gridY];
    }

    bool findPointGridInMap(grid_t *gridOut, PointType *point){
        // Calculate the cube index that this point belongs to. (Array dimension: mapArrayLength * mapArrayLength)
        grid_t thisGrid;
        getPointCubeIndex(&thisGrid.cubeX, &thisGrid.cubeY, point);
        // Decide whether a point is out of pre-allocated map
        if (thisGrid.cubeX >= 0 && thisGrid.cubeX < mapArrayLength && 
            thisGrid.cubeY >= 0 && thisGrid.cubeY < mapArrayLength){
            // Point is in the boundary, but this sub-map is not allocated before
            // Allocate new memory for this sub-map and save it to mapArray
            if (mapArrayInd[thisGrid.cubeX][thisGrid.cubeY] == -1){
                childMap_t *thisChildMap = new childMap_t(mapArrayCount, thisGrid.cubeX, thisGrid.cubeY);
                mapArray.push_back(thisChildMap);
                mapArrayInd[thisGrid.cubeX][thisGrid.cubeY] = mapArrayCount;
                ++mapArrayCount;
            }
        }else{
            // Point is out of pre-allocated boundary, report error (you should increase map size)
            ROS_ERROR("Point cloud is out of elevation map boundary. Change params ->mapArrayLength<-. The program will crash!");
            return false;
        }
        // sub-map id
        thisGrid.mapID = mapArrayInd[thisGrid.cubeX][thisGrid.cubeY];
        // Find the index for this point in this sub-map (grid index)
        thisGrid.gridX = (int)((point->x - mapArray[thisGrid.mapID]->originX) / mapResolution);
        thisGrid.gridY = (int)((point->y - mapArray[thisGrid.mapID]->originY) / mapResolution);
        if (thisGrid.gridX < 0 || thisGrid.gridY < 0 || thisGrid.gridX >= mapCubeArrayLength || thisGrid.gridY >= mapCubeArrayLength)
            return false;

        *gridOut = thisGrid;
        return true;
    }

    void getPointCubeIndex(int *cubeX, int *cubeY, PointType *point){
        *cubeX = int((point->x + mapCubeLength/2.0) / mapCubeLength) + rootCubeIndex;
        *cubeY = int((point->y + mapCubeLength/2.0) / mapCubeLength) + rootCubeIndex;

        if (point->x + mapCubeLength/2.0 < 0)  --*cubeX;
        if (point->y + mapCubeLength/2.0 < 0)  --*cubeY;
    }


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////// Traversability Calculation ///////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    void TraversabilityThread(){

        ros::Rate rate(10); // Hz
        
        while (ros::ok()){

            traversabilityMapCalculation();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            // rate.sleep();
        }
    }
    

    // * compute traversability on cell-level
    // * 3 metrics are considered: elevation difference, slope, roughness
    void traversabilityMapCalculation(){

        // no new scan, return
        if (observingList1.size() == 0) {
            // ROS_INFO_STREAM("RETURN!");
            return;
        }
        // ROS_INFO_STREAM("Traversability Calculating!");
        //对于每一帧观测到的点云，检查是否在过往被多次观测到，若是则计算traversability（问题在于内圈的点可能没有被观测到但是周围的elevation信息发生了改变会导致其traversability改变？）
        observingList2 = observingList1;
        observingList1.clear();

        int listSize = observingList2.size();

        for (int i = 0; i < listSize; ++i){

            mapCell_t *thisCell = observingList2[i];
            // convert this cell to a point for convenience
            PointType thisPoint;
            thisPoint.x = thisCell->xyz->x;
            thisPoint.y = thisCell->xyz->y;
            thisPoint.z = thisCell->xyz->z;
            // too far, not accurate
            if (pointDistance(thisPoint, robotPoint) >= traversabilityCalculatingDistance)
                continue;
            // Find neighbor cells of this center cell
            vector<float> xyzVector = findNeighborElevations(thisCell);
            
            if (xyzVector.size() <= 2)
                continue;

            // matPoints: n * 3 matrix
            // n 对应取的周围的栅格个数，3对应xyz
            Eigen::MatrixXf matPoints = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(xyzVector.data(), xyzVector.size() / 3, 3);
            
            /*
            // min and max elevation
            float minElevation = matPoints.col(2).minCoeff();
            float maxElevation = matPoints.col(2).maxCoeff();
            float maxDifference = maxElevation - minElevation;
            
            if (maxDifference > filterHeightLimit){
                thisPoint.intensity = 100;
                updateCellOccupancy(thisCell, &thisPoint);
                continue;
            }
            
            
            // slope
            Eigen::MatrixXf centered = matPoints.rowwise() - matPoints.colwise().mean(); // 中心化
            Eigen::MatrixXf cov = (centered.adjoint() * centered); //协方差矩阵
            cv::eigen2cv(cov, matCov); // copy data from eigen to cv::Mat
            cv::eigen(matCov, matEig, matVec); // find eigenvalues and eigenvectors for the covariance matrix
            float slopeAngle = std::acos(std::abs(matVec.at<float>(2, 2))) / M_PI * 180;
            
            if (slopeAngle > filterAngleLimit) {
                thisPoint.intensity = 100;
                updateCellOccupancy(thisCell, &thisPoint);
                continue;
            }

            
            
            // roughness
            Eigen::Vector3f norm(matVec.at<float>(2,0),matVec.at<float>(2,1),matVec.at<float>(2,2)); // normal vector
            Eigen::Vector3f mean = matPoints.colwise().mean();
            float planeParam = mean.transpose() * norm;
            Eigen::MatrixXf res = matPoints * norm;
            res.array() -= planeParam;
            double roughness = sqrt(res.squaredNorm() / (xyzVector.size() - 1));
            double roughnessThresh_ = 0.01;
            
            if (roughness > roughnessThresh_) {
                thisPoint.intensity = 100;
                updateCellOccupancy(thisCell, &thisPoint);    
            }
            */
            
        }
    }
    

    
    vector<float> findNeighborElevations(mapCell_t *centerCell){

        vector<float> xyzVector;

        grid_t centerGrid = centerCell->grid;
        grid_t thisGrid;

        int footprintRadiusLength = int(robotRadius / mapResolution);

        for (int k = -footprintRadiusLength; k <= footprintRadiusLength; ++k){
            for (int l = -footprintRadiusLength; l <= footprintRadiusLength; ++l){
                // skip grids too far
                if (std::sqrt(float(k*k + l*l)) * mapResolution > robotRadius)
                    continue;
                // the neighbor grid
                thisGrid.cubeX = centerGrid.cubeX;
                thisGrid.cubeY = centerGrid.cubeY;
                thisGrid.gridX = centerGrid.gridX + k;
                thisGrid.gridY = centerGrid.gridY + l;
                // If the checked grid is in another sub-map, update it's indexes
                if(thisGrid.gridX < 0){ --thisGrid.cubeX; thisGrid.gridX = thisGrid.gridX + mapCubeArrayLength;
                }else if(thisGrid.gridX >= mapCubeArrayLength){ ++thisGrid.cubeX; thisGrid.gridX = thisGrid.gridX - mapCubeArrayLength; }
                if(thisGrid.gridY < 0){ --thisGrid.cubeY; thisGrid.gridY = thisGrid.gridY + mapCubeArrayLength;
                }else if(thisGrid.gridY >= mapCubeArrayLength){ ++thisGrid.cubeY; thisGrid.gridY = thisGrid.gridY - mapCubeArrayLength; }
                // If the sub-map that the checked grid belongs to is empty or not
                int mapInd = mapArrayInd[thisGrid.cubeX][thisGrid.cubeY];
                if (mapInd == -1) continue;
                // the neighbor cell
                mapCell_t *thisCell = grid2Cell(&thisGrid);
                // save neighbor cell for calculating traversability
                
                if (thisCell->elevation != -FLT_MAX && !std::isnan(thisCell->xyz->z)){
                    xyzVector.push_back(thisCell->xyz->x);
                    xyzVector.push_back(thisCell->xyz->y);
                    xyzVector.push_back(thisCell->xyz->z);
                }
            }
        }

        return xyzVector;
    }
    

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////// Occupancy Map (local) //////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void publishMap(){
        // Publish Occupancy Grid Map and Elevation Map
        pubCount++;
        if (pubCount > visualizationFrequency){
            pubCount = 1;
            publishLocalMap();
            publishTraversabilityMap();
            // ROS_INFO("x: %.2f, y:%.2f",robotPoint.x,robotPoint.y); 
        }
    }

    
    void publishLocalMap(){

        if (pubOccupancyMapLocal.getNumSubscribers() == 0 &&
            pubOccupancyMapLocalHeight.getNumSubscribers() == 0)
            return;

        // 1.3 Initialize local occupancy grid map to unknown, height to -FLT_MAX
        std::fill(occupancyMap2DHeight.occupancy.data.begin(), occupancyMap2DHeight.occupancy.data.end(), -1);
        std::fill(occupancyMap2DHeight.height.begin(), occupancyMap2DHeight.height.end(), -FLT_MAX);
        std::fill(occupancyMap2DHeight.costMap.begin(), occupancyMap2DHeight.costMap.end(), 0);
        
        // local map origin x and y （让机器人位置位于local map的中心）
        localMapOriginPoint.x = 83 - localMapLength / 2;
        localMapOriginPoint.y =  104 - localMapLength / 2;
        // localMapOriginPoint.x = robotPoint.x - localMapLength / 2;
        // localMapOriginPoint.y = robotPoint.y - localMapLength / 2;
        localMapOriginPoint.z = robotPoint.z;
        // local map origin cube id (in global map)
        localMapOriginGrid.cubeX = int((localMapOriginPoint.x + mapCubeLength/2.0) / mapCubeLength) + rootCubeIndex;
        localMapOriginGrid.cubeY = int((localMapOriginPoint.y + mapCubeLength/2.0) / mapCubeLength) + rootCubeIndex;
        if (localMapOriginPoint.x + mapCubeLength/2.0 < 0)  --localMapOriginGrid.cubeX;
        if (localMapOriginPoint.y + mapCubeLength/2.0 < 0)  --localMapOriginGrid.cubeY;
        // local map origin grid id (in sub-map)
        float originCubeOriginX, originCubeOriginY; // the orign of submap that the local map origin belongs to (note the submap may not be created yet, cannot use originX and originY)
        originCubeOriginX = (localMapOriginGrid.cubeX - rootCubeIndex) * mapCubeLength - mapCubeLength/2.0;
        originCubeOriginY = (localMapOriginGrid.cubeY - rootCubeIndex) * mapCubeLength - mapCubeLength/2.0;
        localMapOriginGrid.gridX = int((localMapOriginPoint.x - originCubeOriginX) / mapResolution);
        localMapOriginGrid.gridY = int((localMapOriginPoint.y - originCubeOriginY) / mapResolution);

        // 2 Calculate local occupancy grid map root position
        occupancyMap2DHeight.header.stamp = ros::Time::now();
        occupancyMap2DHeight.occupancy.header.stamp = occupancyMap2DHeight.header.stamp;
        occupancyMap2DHeight.occupancy.info.origin.position.x = localMapOriginPoint.x;
        occupancyMap2DHeight.occupancy.info.origin.position.y = localMapOriginPoint.y;
        occupancyMap2DHeight.occupancy.info.origin.position.z = localMapOriginPoint.z + 10; // add 10, just for visualization

        // extract all info
        for (int i = 0; i < localMapArrayLength; ++i){
            for (int j = 0; j < localMapArrayLength; ++j){

                int indX = localMapOriginGrid.gridX + i;
                int indY = localMapOriginGrid.gridY + j;

                grid_t thisGrid;

                thisGrid.cubeX = localMapOriginGrid.cubeX + indX / mapCubeArrayLength;
                thisGrid.cubeY = localMapOriginGrid.cubeY + indY / mapCubeArrayLength;

                thisGrid.gridX = indX % mapCubeArrayLength;
                thisGrid.gridY = indY % mapCubeArrayLength;

                // if sub-map is not created yet
                if (mapArrayInd[thisGrid.cubeX][thisGrid.cubeY] == -1) {
                    continue;
                }
                
                mapCell_t *thisCell = grid2Cell(&thisGrid);

                // skip unknown grid
                if (thisCell->elevation != -FLT_MAX){
                    int index = i + j * localMapArrayLength; // index of the 1-D array 
                    occupancyMap2DHeight.height[index] = thisCell->elevation;
                    occupancyMap2DHeight.occupancy.data[index] = thisCell->occupancy > 80 ? 100 : 0;
                    // geo_OccupancyMap2D.occupancy.data[index] = thisCell->geo_occupancy > 80 ? 100 : 0;
                    // sem_OccupancyMap2D.occupancy.data[index] = thisCell->sem_occupancy > 80 ? 100 : 0;
                }
            }
        }

        pubOccupancyMapLocalHeight.publish(occupancyMap2DHeight);
        pubOccupancyMapLocal.publish(occupancyMap2DHeight.occupancy);
        // pubGeoOccupancyMapLocal.publish(geo_OccupancyMap2D);
        // pubSemOccupancyMapLocal.publish(sem_OccupancyMap2D);
    }
    
    

    void initializeLocalOccupancyMap(){
        // initialization of customized map message
        occupancyMap2DHeight.header.frame_id = ns+"/map";
        occupancyMap2DHeight.occupancy.info.width = localMapArrayLength;
        occupancyMap2DHeight.occupancy.info.height = localMapArrayLength;
        occupancyMap2DHeight.occupancy.info.resolution = mapResolution;
        
        occupancyMap2DHeight.occupancy.info.origin.orientation.x = 0.0;
        occupancyMap2DHeight.occupancy.info.origin.orientation.y = 0.0;
        occupancyMap2DHeight.occupancy.info.origin.orientation.z = 0.0;
        occupancyMap2DHeight.occupancy.info.origin.orientation.w = 1.0;

        occupancyMap2DHeight.occupancy.data.resize(occupancyMap2DHeight.occupancy.info.width * occupancyMap2DHeight.occupancy.info.height);
        occupancyMap2DHeight.height.resize(occupancyMap2DHeight.occupancy.info.width * occupancyMap2DHeight.occupancy.info.height);
        occupancyMap2DHeight.costMap.resize(occupancyMap2DHeight.occupancy.info.width * occupancyMap2DHeight.occupancy.info.height);
    }    

    bool getRobotPosition(){
        try{listener.lookupTransform(ns+"/map",ns+"/base_link", ros::Time(0), transform); }  // lookupTransform(target_frame, source_frame)  base_link 到 map 的变换，即map坐标系下的xyz坐标
        catch (tf::TransformException ex){ ROS_ERROR("Transfrom Failure in Traversaility Map."); return false; }

        robotPoint.x = transform.getOrigin().x();
        robotPoint.y = transform.getOrigin().y();
        robotPoint.z = transform.getOrigin().z();

        return true;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////// Point Cloud /////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void publishTraversabilityMap(){

        if (pubElevationCloud.getNumSubscribers() == 0)
            return;
        // 1. Find robot current cube index
        int currentCubeX, currentCubeY;
        getPointCubeIndex(&currentCubeX, &currentCubeY, &robotPoint);
        // 2. Loop through all the sub-maps that are nearby
        int visualLength = int(visualizationRadius / mapCubeLength);
        // int visualLength = 1; // 用于建全局可通行地图
        for (int i = -visualLength; i <= visualLength; ++i){
            for (int j = -visualLength; j <= visualLength; ++j){

                if (sqrt(float(i*i+j*j)) >= visualLength) continue;  // 半径50米以内的圆

                int idx = i + currentCubeX;
                int idy = j + currentCubeY;

                if (idx < 0 || idx >= mapArrayLength ||  idy < 0 || idy >= mapArrayLength) continue;

                if (mapArrayInd[idx][idy] == -1) continue;

                *laserCloudElevation += mapArray[mapArrayInd[idx][idy]]->cloud;
            }
        }
        // 3. Publish elevation point cloud
        sensor_msgs::PointCloud2 laserCloudTemp;
        pcl::toROSMsg(*laserCloudElevation, laserCloudTemp);
        laserCloudTemp.header.frame_id = ns+"/map";
        laserCloudTemp.header.stamp = ros::Time::now();
        pubElevationCloud.publish(laserCloudTemp);
        // 4. free memory
        laserCloudElevation->clear();
    }
};




int main(int argc, char** argv){

    ros::init(argc, argv, "traversability_mapping");
    
    TraversabilityMapping tMapping;

    // 开辟一个线程，在tMapping对象上运行其类成员函数 TraversabilityMapping::TraversabilityThread
    std::thread predictionThread(&TraversabilityMapping::TraversabilityThread, &tMapping); 

    ROS_INFO("\033[1;32m---->\033[0m Traversability Mapping Started.");
    ROS_INFO("\033[1;32m---->\033[0m Traversability Mapping Scenario: %s.", 
        urbanMapping == true ? "\033[1;31mUrban\033[0m" : "\033[1;31mTerrain\033[0m");

    // ros::AsyncSpinner asyncSpinner(2);
    // asyncSpinner.start();
    // ros::waitForShutdown();
    ros::spin();

    return 0;
}