%%%%hpp
cv::Matx33d m_R;		// rotation (p' = m_R*p + m_t) // initialized by Identity
cv::Matx31d m_t;		// translation (p' = m_R*p + m_t) // initialized by Zero

%%%%cpp
#define RAD2DEG(x)         ((x) * 180.0 / CV_PI)
#define DEG2RAD(x)         ((x) * CV_PI / 180.0)

%%%% File load & param load
bool loadConfig(const cv::FileNode& fn)
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
        }
        else return false;
    }
}

void SE3::setPosePanTiltRoll(double x, double y, double z, double pan, double tilt, double roll)
{
    setPanTiltRollPinned(pan, tilt, roll);
    setPosition(x, y, z);
}
void SE3::setPanTiltRollPinned(double pan, double tilt, double roll)
{
    Matx33d R = rotationFromPanTiltRoll(pan, tilt, roll);
    setRotationPinned(R);
}

Matx33d SE3::rotationFromPanTiltRoll(double pan, double tilt, double roll)
{
    Matx33d R;
    R(0,0) = sin(pan)*cos(roll)-cos(pan)*sin(tilt)*sin(roll);
    R(0,1) = -cos(pan)*cos(roll)-sin(pan)*sin(tilt)*sin(roll);
    R(0,2) = cos(tilt)*sin(roll);
    R(1,0) = sin(pan)*sin(roll)+sin(tilt)*cos(pan)*cos(roll);
    R(1,1) = -cos(pan)*sin(roll)+sin(tilt)*sin(pan)*cos(roll);
    R(1,2) = -cos(tilt)*cos(roll);
    R(2,0) = cos(tilt)*cos(pan);
    R(2,1) = cos(tilt)*sin(pan);
    R(2,2) = sin(tilt);

    return R;
}
void SE3::setRotationPinned(const Matx33d& R)
{
    m_t = R*m_R.t()*m_t;
    m_R = R;
}
void SE3::setPosition(double x, double y, double z)
{
    Matx31d p(x,y,z);

    m_t = -m_R*p;
}
void SE3::getRt(Matx33d& R, Matx31d& t) const
{
    R = m_R;
    t = m_t;
}

void SE3::getPosition(double& x, double& y, double& z) const
{
    Matx31d p = -m_R.t()*m_t;

    x = p(0);
    y = p(1);
    z = p(2);
}


############## main ######################
loadConfig();

cv::Matx33d R, Rinv;
cv::Matx31d t;
m_se3.getRt(R, t);
Rinv = R.t();

double x, y, z;
m_se3.getPosition(x, y, z);

dst.resize(src.size());
valid.resize(src.size());

for (int i = 0; i < (int)src.size(); i++)
{
cv::Point2d n;
if (m_distortion_model_type == DM_COUPLED)
{
    // normalized image coordinate
    n.x = (src[i].x - m_intrinsic.cx) / m_intrinsic.fx;
    n.y = (src[i].y - m_intrinsic.cy) / m_intrinsic.fy;

    // correct distortion
    _undistort(n);
}
else
{
    // centered image coordinate
    cv::Point2d src_undistorted;
    src_undistorted.x = src[i].x - m_intrinsic.cx;
    src_undistorted.y = src[i].y - m_intrinsic.cy;

    // correct distortion
    _undistort(src_undistorted);

    // normalized image coordinate
    n.x = src_undistorted.x / m_intrinsic.fx;
    n.y = src_undistorted.y / m_intrinsic.fy;
}

// world coordinate of normalized image coordinate
cv::Matx31d Pc(n.x, n.y, 1);
cv::Matx31d Pw = Rinv * (Pc - t);

double px = Pw(0);
double py = Pw(1);
double pz = Pw(2);


