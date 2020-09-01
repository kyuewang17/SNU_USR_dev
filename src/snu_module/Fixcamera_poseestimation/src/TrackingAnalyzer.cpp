#include "TrackingAnalyzer.hpp"

#define RAD2DEG(x)         ((x) * 180.0 / CV_PI)
#define DEG2RAD(x)         ((x) * CV_PI / 180.0)

#define LOAD_PARAM_VALUE(fn, name_cfg, name_var) \
    if (!(fn)[name_cfg].empty()) (fn)[name_cfg] >> name_var;

#define CAMERA_PARAM_SIZE 11


bool TrackingAnalyzer::initialize(std::string camera_id, std::string config_file)
{
    m_time_stamp = 0;
    m_geofilter.setCamera(NULL);
    return loadConfigFile(camera_id, config_file);
}

bool TrackingAnalyzer::update(const std::vector<bbox_t>& boxes, double image_acquisition_time_sec)
{
    // pedestrian(bird, dog) & vehicle detections
    std::vector<cv::Rect> rc_persons;
    std::vector<cv::Rect> rc_vehicles;
    for (size_t i = 0; i < boxes.size(); i++)
    //for (size_t i = 0; i < num; i++)
    {
        if (boxes[i].prob < m_param.yolo_threshold) continue;

        unsigned int id = boxes[i].obj_id;
        if (id == 0 || id == 14 || id == 16)    // person, bird, dog
            rc_persons.push_back(cv::Rect(boxes[i].x, boxes[i].y, boxes[i].w, boxes[i].h));
        else if (id == 2)    // car
            rc_vehicles.push_back(cv::Rect(boxes[i].x, boxes[i].y, boxes[i].w, boxes[i].h));
    }

    // filtering
    if(m_param.enable_geofilter) m_geofilter.correct(rc_persons);

    // tracking
    m_tracker_people.update(rc_persons, image_acquisition_time_sec);
    m_tracker_vehicle.update(rc_vehicles, image_acquisition_time_sec);

    // tracking info of the detected objects
    std::vector<box_t>& objects_person = m_tracker_people.objects();
    std::vector<box_t>& objects_vehicle = m_tracker_vehicle.objects();
    int n_person = objects_person.size();
    int n_vehicle = objects_vehicle.size();
    std::vector<int> class_id;

    std::vector<cv::Point2d> pts, pts_prev;
    std::vector<int> track_id;
    std::vector<double> delta_t;
    std::vector<cv::Rect> track_rc;
    std::vector<cv::Scalar> track_color;
    for (int i = 0; i < n_person + n_vehicle; i++)
    {
        box_t obj = (i < n_person) ? objects_person[i] : objects_vehicle[i - n_person];
        if (!obj.detected) continue;

        if (i < n_person) class_id.push_back(0);
        else class_id.push_back(2);

        cv::Point2f pt, pt_prev;
        int n = (int)obj.smoothed_pts.size();
        pt.x = obj.smoothed_pts[n - 1].x;
        pt.y = obj.smoothed_pts[n - 1].y + obj.smoothed_rc_h[n - 1] / 2;
        pt_prev = pt;
        if (n >= 2)
        {
            pt_prev.x = obj.smoothed_pts[n - 2].x;
            pt_prev.y = obj.smoothed_pts[n - 2].y + obj.smoothed_rc_h[n - 2] / 2;
        }
        pts.push_back(pt);
        pts_prev.push_back(pt_prev);

        delta_t.push_back((n>=2) ? obj.time_stamp[n - 1] - obj.time_stamp[n - 2] : 0);
        track_id.push_back(obj.id);
        cv::Rect rc = obj.rc;
        track_rc.push_back(rc);
        track_color.push_back(obj.color);
    }

    // world coordinates of the detected pedestrians
    std::vector<cv::Point2d> world, world_prev;
    transformPointsImage2World(pts, world);
    transformPointsImage2World(pts_prev, world_prev);

    // record data
    m_result.clear();
    for (size_t i = 0; i < world.size(); i++)
    {
        double dt = delta_t[i];
        if (dt <= 0) dt = 1;

        obj_t obj;
        obj.class_id = class_id[i];
        obj.obj_id = track_id[i];
        obj.x = (int)(world[i].x * 100 + 0.5); // centimeter
        obj.y = (int)(world[i].y * 100 + 0.5); // centimeter
        obj.vx = (int)((world[i].x - world_prev[i].x) * 100 / dt + 0.5); // centimeter/sec
        obj.vy = (int)((world[i].y - world_prev[i].y) * 100 / dt + 0.5); // centimeter/sec
        obj.ix = track_rc[i].x;
        obj.iy = track_rc[i].y;
        obj.iw = track_rc[i].width;
        obj.ih = track_rc[i].height;
        obj.color = track_color[i];

        m_result.push_back(obj);
    }

    m_time_stamp = image_acquisition_time_sec;
}


bool TrackingAnalyzer::loadConfigFile(std::string camera_id, std::string config_file)
{
    if (config_file.empty()) return false;

    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;

    // Load default parameters defined in default FileNode
    cv::FileNode fn_default = fs["DEFAULT"];
    if (!fn_default.empty()) loadConfig(fn_default);

    // Load parameters defined in top-level
    loadConfig(fs.root());

    // Load camera-specific parameters
    cv::FileNode fn_specific = fs[camera_id];
    if (!fn_specific.empty()) loadConfig(fn_specific);

    return true;
}


bool TrackingAnalyzer::loadConfig(const cv::FileNode& fn)
{
    // YOLO Parameters
    LOAD_PARAM_VALUE(fn, "yolo_threshold", m_param.yolo_threshold);

    // Camera Parameters
    cv::FileNode camera_cfg = (fn)["camera_param"];
    if (!camera_cfg.empty())
    {
        std::vector<double> camera_param;
        camera_cfg >> camera_param;
        if (camera_param.size() == CAMERA_PARAM_SIZE)
        {
            evl::CameraParam param;
            param.fx = camera_param[0];
            param.fy = camera_param[1];
            param.cx = camera_param[2];
            param.cy = camera_param[3];
            param.w = camera_param[4];
            m_camera.set_intrinsic_parameters(param);

            evl::SE3 se3;
            double x = camera_param[5];
            double y = camera_param[6];
            double z = camera_param[7];
            double pan = DEG2RAD(camera_param[8]);
            double tilt = DEG2RAD(camera_param[9]);
            double roll = DEG2RAD(camera_param[10]);
            se3.setPosePanTiltRoll(x, y, z, pan, tilt, roll);
            m_camera.set_extrinsic(se3);

            m_geofilter.setCamera(&m_camera);
        }
        else return false;
    }

    // Geometric Filter Parameters
    LOAD_PARAM_VALUE(fn, "enable_geofilter", m_param.enable_geofilter);
    if (m_param.enable_geofilter) m_geofilter.loadParam(fn);

    // BoxTracker Parameters
    m_tracker_people.loadParam(fn);
    m_tracker_vehicle.loadParam(fn);
}

void TrackingAnalyzer::transformPointsImage2World(const std::vector<cv::Point2d>& image_pts, std::vector<cv::Point2d>& world_pts)
{
    if (image_pts.size() <= 0) return;
    std::vector<bool> valid;
    m_camera.unproject_ground(image_pts, world_pts, valid);
}