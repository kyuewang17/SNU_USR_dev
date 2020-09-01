#include "virtual_camera.hpp"
#include "opencv/cxcore.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"

#define RAD2DEG(x)         ((x) * 180.0 / CV_PI)
#define DEG2RAD(x)         ((x) * CV_PI / 180.0)

namespace evl
{
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    CameraBase::CameraBase()
    {
        m_distortion_model_type = DM_COUPLED;
    }

    void CameraBase::set_coupled_distortion_model()
    {
        m_distortion_model_type = DM_COUPLED;
    }

    void CameraBase::set_decoupled_distortion_model()
    {
        m_distortion_model_type = DM_DECOUPLED;
    }

    bool CameraBase::load_evlcamera(const char* file_path, const char* cam_id)
    {
        // load data file
        cv::FileStorage fs(file_path, cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            printf("Fail to open %s\n", file_path);
            return false;
        }

        // read camera info
        std::vector<double> cam_param;
        fs[cam_id] >> cam_param;
        if (cam_param.size() != 11)
        {
            printf("Fail to retrieve %s info\n", cam_id);
            return false;
        }

        m_intrinsic.fx = cam_param[0];
        m_intrinsic.fy = cam_param[1];
        m_intrinsic.cx = cam_param[2];
        m_intrinsic.cy = cam_param[3];
        m_intrinsic.w = cam_param[4];

        double x = cam_param[5];
        double y = cam_param[6];
        double z = cam_param[7];
        double pan = DEG2RAD(cam_param[8]);
        double tilt = DEG2RAD(cam_param[9]);
        double roll = DEG2RAD(cam_param[10]);
        m_se3.setPosePanTiltRoll(x, y, z, pan, tilt, roll);
        return true;
    }

    void CameraBase::set_intrinsic_parameters(const CameraParam& param)
    {
        m_intrinsic = param;
    }

    CameraParam CameraBase::get_intrinsic_parameters() const
    {
        return m_intrinsic;
    }

    void CameraBase::set_extrinsic(const SE3& se3)
    {
        m_se3 = se3;
    }

    SE3 CameraBase::get_extrinsic() const
    {
        return m_se3;
    }

    bool CameraBase::project(const cv::Point3d& src, cv::Point2d& dst) const
    {
        double depth;
        return project(src, dst, depth);
    }

    bool CameraBase::project(const cv::Point3d& src, cv::Point2d& dst, double& depth) const
    {
        cv::Matx33d R;
        cv::Matx31d t;
        m_se3.getRt(R, t);

        // camera coordinate
        cv::Matx31d Pw(src.x, src.y, src.z);
        cv::Matx31d Pc = R*Pw + t;

        double Xc = Pc(0);
        double Yc = Pc(1);
        depth = Pc(2);
        double eps = 1e-10;
        if (fabs(depth) <= eps)
        {
            dst.x = -1;
            dst.y = -1;
            depth = 0;
            return false;
        }

        // normalized image coordinate
        cv::Point2d n(Xc / depth, Yc / depth);

        if (m_distortion_model_type == DM_COUPLED)
        {
            // distorted normalized image coordinate
            _distort(n);

            // pixel coordinate
            dst.x = n.x * m_intrinsic.fx + m_intrinsic.cx;
            dst.y = n.y * m_intrinsic.fy + m_intrinsic.cy;
        }
        else
        {
            // centered pixel coordinate
            dst.x = n.x * m_intrinsic.fx ;
            dst.y = n.y * m_intrinsic.fy;

            // distorted centered image coordinate
            _distort(dst);

            // decentered pixel coordinate
            dst.x = dst.x + m_intrinsic.cx;
            dst.y = dst.y + m_intrinsic.cy;
        }

        return true;
    }

    bool CameraBase::unproject_ground(const cv::Point2d& src, cv::Point2d& dst) const
    {
        cv::Point2d n;
        if (m_distortion_model_type == DM_COUPLED)
        {
            // normalized image coordinate
            n.x = (src.x - m_intrinsic.cx) / m_intrinsic.fx;
            n.y = (src.y - m_intrinsic.cy) / m_intrinsic.fy;

            // correct distortion
            _undistort(n);
        }
        else
        {
            // centered image coordinate
            cv::Point2d src_undistorted;
            src_undistorted.x = src.x - m_intrinsic.cx;
            src_undistorted.y = src.y - m_intrinsic.cy;

            // correct distortion
            _undistort(src_undistorted);

            // normalized image coordinate
            n.x = (src_undistorted.x) / m_intrinsic.fx;
            n.y = (src_undistorted.y) / m_intrinsic.fy;
        }

        // world coordinate of normalized image coordinate
        cv::Matx33d R, Rinv;
        cv::Matx31d t;
        m_se3.getRt(R, t);
        Rinv = R.t();
        cv::Matx31d Pc(n.x, n.y, 1);
        cv::Matx31d Pw = Rinv * (Pc - t);

        double px = Pw(0);
        double py = Pw(1);
        double pz = Pw(2);

        double x, y, z;
        m_se3.getPosition(x, y, z);

        // check validity (if the ray meets the ground or not)
        double eps = 1e-10;
        if (pz >= z - eps) return false;

        // project to world ground plane
        double k = z / (z - pz);

        dst.x = x + k*(px - x);
        dst.y = y + k*(py - y);

        return true;
    }

    bool CameraBase::unproject_ground_relative(const cv::Point2d& src, double& distance, double& theta_radian) const
    {
        cv::Point2d n;
        if (m_distortion_model_type == DM_COUPLED)
        {
            // normalized image coordinate
            n.x = (src.x - m_intrinsic.cx) / m_intrinsic.fx;
            n.y = (src.y - m_intrinsic.cy) / m_intrinsic.fy;

            // correct distortion
            _undistort(n);
        }
        else
        {
            // centered image coordinate
            cv::Point2d src_undistorted;
            src_undistorted.x = src.x - m_intrinsic.cx;
            src_undistorted.y = src.y - m_intrinsic.cy;

            // correct distortion
            _undistort(src_undistorted);

            // normalized image coordinate
            n.x = (src_undistorted.x) / m_intrinsic.fx;
            n.y = (src_undistorted.y) / m_intrinsic.fy;
        }

        // world coordinate of normalized image coordinate
        cv::Matx33d R, Rinv;
        cv::Matx31d t;
        m_se3.getRt(R, t);
        Rinv = R.t();
        cv::Matx31d Pc(n.x, n.y, 1);
        cv::Matx31d Pw = Rinv * (Pc - t);

        double px = Pw(0);
        double py = Pw(1);
        double pz = Pw(2);

        double x, y, z, pan, tilt, roll;
        m_se3.getPosePanTiltRoll(x, y, z, pan, tilt, roll);

        // check validity (if the ray meets the ground or not)
        double eps = 1e-10;
        if (pz >= z - eps) return false;

        // project to world ground plane
        double k = z / (z - pz);
        double dx = x + k*(px - x);
        double dy = y + k*(py - y);
        double dpan = atan2(dy, dx);

        distance = sqrt((dx - x)*(dx - x) + (dy - y)*(dy - y));
        theta_radian = dpan - pan;
        while (theta_radian > CV_PI) theta_radian -= 2 * CV_PI;
        while (theta_radian <= -CV_PI) theta_radian += 2 * CV_PI;

        return true;
    }

    void CameraBase::project(const std::vector<cv::Point3d>& src, std::vector<cv::Point2d>& dst) const
    {
        std::vector<double> depth;
        return project(src, dst, depth);
    }

    void CameraBase::project(const std::vector<cv::Point3d>& src, std::vector<cv::Point2d>& dst, std::vector<double>& depth) const
    {
        cv::Matx33d R;
        cv::Matx31d t;
        m_se3.getRt(R, t);

        dst.resize(src.size());
        depth.resize(src.size());

        double eps = 1e-10;
        for (int i = 0; i < (int)src.size(); i++)
        {
            // camera coordinate
            cv::Matx31d Pw(src[i].x, src[i].y, src[i].z);
            cv::Matx31d Pc = R*Pw + t;

            depth[i] = Pc(2);
            if (fabs(depth[i]) <= eps)
            {
                dst[i].x = -1;
                dst[i].y = -1;
                depth[i] = 0;
                continue;
            }

            // normalized image coordinate
            double Xc = Pc(0);
            double Yc = Pc(1);
            cv::Point2d n(Xc / depth[i], Yc / depth[i]);

            if (m_distortion_model_type == DM_COUPLED)
            {
                // distorted normalized image coordinate
                _distort(n);

                // pixel coordinate
                dst[i].x = n.x * m_intrinsic.fx + m_intrinsic.cx;
                dst[i].y = n.y * m_intrinsic.fy + m_intrinsic.cy;
            }
            else
            {
                // centered pixel coordinate
                dst[i].x = n.x * m_intrinsic.fx;
                dst[i].y = n.y * m_intrinsic.fy;

                // distorted centered image coordinate
                _distort(dst[i]);

                // decentered pixel coordinate
                dst[i].x = dst[i].x + m_intrinsic.cx;
                dst[i].y = dst[i].y + m_intrinsic.cy;
            }
        }
    }

    void CameraBase::unproject_ground(const std::vector<cv::Point2d>& src, std::vector<cv::Point2d>& dst, std::vector<bool>& valid) const
    {
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

            // check validity (if the ray meets the ground or not)
            double eps = 1e-10;
            if (pz >= z - eps)
            {
                dst[i].x = 0;
                dst[i].y = 0;
                valid[i] = false;
                continue;
            }

            // project to world ground plane
            double k = z / (z - pz);

            dst[i].x = x + k*(px - x);
            dst[i].y = y + k*(py - y);
            valid[i] = true;
        }
    }

    void CameraBase::unproject_ground_relative(const std::vector<cv::Point2d>& src, std::vector<double>& distance, std::vector<double>& theta_radian, std::vector<bool>& valid) const
    {
        cv::Matx33d R, Rinv;
        cv::Matx31d t;
        m_se3.getRt(R, t);
        Rinv = R.t();

        double x, y, z, pan, tilt, roll;
        m_se3.getPosePanTiltRoll(x, y, z, pan, tilt, roll);

        distance.resize(src.size());
        theta_radian.resize(src.size());
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

            // check validity (if the ray meets the ground or not)
            double eps = 1e-10;
            if (pz >= z - eps)
            {
                distance[i] = 0;
                theta_radian[i] = 0;
                valid[i] = false;
                continue;
            }

            // project to world ground plane
            double k = z / (z - pz);
            double dx = x + k*(px - x);
            double dy = y + k*(py - y);
            double dpan = atan2(dy, dx);

            distance[i] = sqrt((dx - x)*(dx - x) + (dy - y)*(dy - y));
            theta_radian[i] = dpan - pan;
            while (theta_radian[i] > CV_PI) theta_radian[i] -= 2 * CV_PI;
            while (theta_radian[i] <= -CV_PI) theta_radian[i] += 2 * CV_PI;
            valid[i] = true;
        }
    }

    void CameraBase::normalize(cv::Point2d& pt) const
    {
        pt.x = (pt.x - m_intrinsic.cx) / m_intrinsic.fx;
        pt.y = (pt.y - m_intrinsic.cy) / m_intrinsic.fy;
    }

    void CameraBase::normalize(std::vector<cv::Point2d>& pts) const
    {
        for (int i = 0; i < (int)pts.size(); i++)
        {
            pts[i].x = (pts[i].x - m_intrinsic.cx) / m_intrinsic.fx;
            pts[i].y = (pts[i].y - m_intrinsic.cy) / m_intrinsic.fy;
        }
    }

    void CameraBase::denormalize(cv::Point2d& pt) const
    {
        pt.x = pt.x * m_intrinsic.fx + m_intrinsic.cx;
        pt.y = pt.y * m_intrinsic.fy + m_intrinsic.cy;
    }

    void CameraBase::denormalize(std::vector<cv::Point2d>& pts) const
    {
        for (int i = 0; i < (int)pts.size(); i++)
        {
            pts[i].x = pts[i].x * m_intrinsic.fx + m_intrinsic.cx;
            pts[i].y = pts[i].y * m_intrinsic.fy + m_intrinsic.cy;
        }
    }

    void CameraBase::distort(cv::Point2d& pt) const
    {
        if (m_distortion_model_type == DM_COUPLED)
        {
            normalize(pt);
            _distort(pt);
            denormalize(pt);
        }
        else
        {
            _center_normalize(pt);
            _distort(pt);
            _center_denormalize(pt);
        }
    }

    void CameraBase::distort(std::vector<cv::Point2d>& pts) const
    {
        if (m_distortion_model_type == DM_COUPLED)
        {
            normalize(pts);
            _distort(pts);
            denormalize(pts);
        }
        else
        {
            _center_normalize(pts);
            _distort(pts);
            _center_denormalize(pts);
        }
    }

    void CameraBase::undistort(cv::Point2d& pt) const
    {
        if (m_distortion_model_type == DM_COUPLED)
        {
            normalize(pt);
            _undistort(pt);
            denormalize(pt);
        }
        else
        {
            _center_normalize(pt);
            _undistort(pt);
            _center_denormalize(pt);
        }
    }

    void CameraBase::undistort(std::vector<cv::Point2d>& pts) const
    {
        if (m_distortion_model_type == DM_COUPLED)
        {
            normalize(pts);
            _undistort(pts);
            denormalize(pts);
        }
        else
        {
            _center_normalize(pts);
            _undistort(pts);
            _center_denormalize(pts);
        }
    }

    void CameraBase::distort_image(const cv::Mat& src, cv::Mat& dst) const
    {
        int w = src.cols;
        int h = src.rows;

        std::vector<cv::Point2d> pts(w*h);
        int i = 0;
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                pts[i].x = x;
                pts[i].y = y;
                i++;
            }
        }
        undistort(pts);

        dst = src.clone();
        cv::Rect imgrc(0, 0, w, h);
        if (src.channels() > 1)
        {
            int i = 0;
            for (int y = 0; y < h; y++)
            {
                unsigned char *p = dst.data + y*dst.step;
                for (int x = 0; x < w; x++)
                {
                    cv::Point pt = pts[i];
                    if (pt.inside(imgrc))
                    {
                        unsigned char *q = src.data + pt.y*src.step + pt.x * 3;

                        p[0] = q[0];
                        p[1] = q[1];
                        p[2] = q[2];
                    }
                    i++;
                    p = p + 3;
                }
            }
        }
        else
        {
            int i = 0;
            for (int y = 0; y < h; y++)
            {
                unsigned char *p = dst.data + y*dst.step;
                for (int x = 0; x < w; x++)
                {
                    cv::Point pt = pts[i];
                    if (pt.inside(imgrc))
                    {
                        p[x] = src.data[pt.y*src.step + pt.x];
                    }
                    i++;
                }
            }
        }
    }

    void CameraBase::undistort_image(const cv::Mat& src, cv::Mat& dst, double dst_scale, double dst_size, bool canonical_view)
    {
        _build_lookup_table(src, dst_scale, canonical_view);

        // src size
        int sw = src.cols;
        int sh = src.rows;

        // dst size
        int dw = m_dst.cols;
        int dh = m_dst.rows;

        // undistorted (canonical) image
        cv::Rect imgrc(0, 0, sw, sh);
        if (src.channels() > 1)
        {
            int i = 0;
            for (int y = 0; y < dh; y++)
            {
                unsigned char *p = m_dst.data + y*m_dst.step;
                for (int x = 0; x < dw; x++)
                {
                    if (m_lookup_pts[i].inside(imgrc))
                    {
                        unsigned char *q = src.data + m_lookup_pts[i].y*src.step + m_lookup_pts[i].x * 3;
                        p[0] = q[0];
                        p[1] = q[1];
                        p[2] = q[2];
                    }
                    i++;
                    p = p + 3;
                }
            }
        }
        else
        {
            int i = 0;
            for (int y = 0; y < dh; y++)
            {
                unsigned char *p = m_dst.data + y*m_dst.step;
                for (int x = 0; x < dw; x++)
                {
                    if (m_lookup_pts[i].inside(imgrc))
                    {
                        p[x] = src.data[m_lookup_pts[i].y*src.step + m_lookup_pts[i].x];
                    }
                    i++;
                }
            }
        }

        // resize undistorted image to dst_size
        int dst_w = (int)(sw * dst_size + 0.5);
        int dst_h = (int)(sh * dst_size + 0.5);
        if(dst_size>0 && (m_dst.rows!=dst_h || m_dst.cols!=dst_w)) cv::resize(m_dst, dst, cv::Size(dst_w, dst_h));
        else dst = m_dst;
    }

    bool is_equal(const CameraParam& p1, const CameraParam& p2)
    {
        if (p1.fx != p2.fx) return false;
        if (p1.fy != p2.fy) return false;
        if (p1.cx != p2.cx) return false;
        if (p1.cy != p2.cy) return false;
        if (p1.w != p2.w) return false;
        if (p1.k1 != p2.k1) return false;
        if (p1.k2 != p2.k2) return false;
        if (p1.k3 != p2.k3) return false;
        if (p1.k4 != p2.k4) return false;
        if (p1.p1 != p2.p1) return false;
        if (p1.p2 != p2.p2) return false;
        return true;
    }

    void CameraBase::_build_lookup_table(const cv::Mat& src, double dst_scale, bool canonical_view)
    {
        if (!m_lookup_pts.empty() && dst_scale == m_dst_scale && canonical_view == m_canonical_view && is_equal(m_intrinsic, m_lookup_intrinsic)) return;
        m_dst_scale = dst_scale;
        m_canonical_view = canonical_view;

        // src size
        int sw = src.cols;
        int sh = src.rows;

        // automatic selection of dst scale
        int canonical_offset_y = 0;
        if (dst_scale <= 0)
        {
            std::vector<cv::Point2d> boundary_pts_horz(2);
            boundary_pts_horz[0].x = 0; boundary_pts_horz[0].y = m_intrinsic.cy;
            boundary_pts_horz[1].x = sw - 1; boundary_pts_horz[1].y = m_intrinsic.cy;

            undistort(boundary_pts_horz);
            double dx1 = fabs(boundary_pts_horz[0].x - m_intrinsic.cx);
            double dx2 = fabs(boundary_pts_horz[1].x - m_intrinsic.cx);
            double dx = (dx1 < dx2) ? dx1 : dx2;

            std::vector<cv::Point2d> boundary_pts_vert(2);
            boundary_pts_vert[0].x = m_intrinsic.cx; boundary_pts_vert[0].y = 0;
            boundary_pts_vert[1].x = m_intrinsic.cx; boundary_pts_vert[1].y = sh - 1;

            undistort(boundary_pts_vert);
            double dy1 = fabs(boundary_pts_vert[0].y - m_intrinsic.cy);
            double dy2 = fabs(boundary_pts_vert[1].y - m_intrinsic.cy);
            double dy = (dy1 < dy2) ? dy1 : dy2;

            double x_scale = dx / m_intrinsic.cx;
            double y_scale = dy / m_intrinsic.cy;
            if (canonical_view)
            {
                std::vector<cv::Point2d> boundary_pts;
                boundary_pts.push_back(cv::Point2d(0, m_intrinsic.cy));
                boundary_pts.push_back(cv::Point2d(sw - 1, m_intrinsic.cy));
                boundary_pts.push_back(cv::Point2d(m_intrinsic.cx, 0));
                boundary_pts.push_back(cv::Point2d(m_intrinsic.cx, sh - 1));
                pixel2canonical(boundary_pts);

                double sx = fabs(boundary_pts[1].x - boundary_pts[0].x) / sw;
                double sy = fabs(boundary_pts[3].y - boundary_pts[2].y) / sh;
                if (sx > x_scale) x_scale = sx;
                if (sy > y_scale) y_scale = sy;
                canonical_offset_y = (boundary_pts[3].y + boundary_pts[2].y) / 2 - m_intrinsic.cy;
            }
            dst_scale = (x_scale > y_scale) ? x_scale : y_scale;
        }

        // dst image
        int dw = (int)(sw * dst_scale + 0.5);
        int dh = (int)(sh * dst_scale + 0.5);
        m_dst = cv::Mat::zeros(dh, dw, src.type());

        // dst image points
        std::vector<cv::Point2d> pts(dw*dh);
        int offset_x = (sw - dw) / 2;
        int offset_y = (sh - dh) / 2 + canonical_offset_y;
        int i = 0;
        for (int y = 0; y < dh; y++)
        {
            for (int x = 0; x < dw; x++)
            {
                pts[i].x = x + offset_x;
                pts[i].y = y + offset_y;
                i++;
            }
        }

        // canonical view
        if (canonical_view)
        {
            canonical2pixel(pts);
        }
        else
        {
            distort(pts);
        }

        m_lookup_pts.resize(pts.size());
        for (int k = 0; k < (int)pts.size(); k++)
        {
            m_lookup_pts[k] = pts[k];
        }

        m_lookup_intrinsic = m_intrinsic;
    }


    void CameraBase::spherical_image(const cv::Mat& src, cv::Mat& dst) const
    {
        double img_w = src.cols;
        double img_h = src.rows;

        dst = cv::Mat::zeros(img_h, img_w, src.type());

        double fx = m_intrinsic.fx;
        double fy = m_intrinsic.fy;
        double cx = m_intrinsic.cx;
        double cy = m_intrinsic.cy;

        if (src.channels() > 1)
        {
            for (int yp = 0; yp < img_h; yp++)
            {
                for (int xp = 0; xp < img_w; xp++)
                {
                    double up = (xp - cx) / fx;
                    double vp = (yp - cy) / fy;
                    double rp = sqrt(up*up + vp*vp);
                    if (rp < 0.000001)
                    {
                        dst.at<cv::Vec3b>(yp, xp) = src.at<cv::Vec3b>(cy, cx);
                        continue;
                    }
                    double s = tan(rp) / rp;
                    double x = up*s*fx + cx;
                    double y = vp*s*fy + cy;
                    if (x >= 0 && x < img_w && y >= 0 && y < img_h)
                    {
                        dst.at<cv::Vec3b>(yp, xp) = src.at<cv::Vec3b>(y, x);
                    }
                }
            }
        }
        else
        {
            for (int yp = 0; yp < img_h; yp++)
            {
                unsigned char *p = dst.data + yp*dst.step;
                for (int xp = 0; xp < img_w; xp++)
                {
                    double up = (xp - cx) / fx;
                    double vp = (yp - cy) / fy;
                    double rp = sqrt(up*up + vp*vp);
                    if (rp < 0.000001)
                    {
                        p[xp] = src.at<unsigned char>(cy, cx);
                        continue;
                    }
                    double s = tan(rp) / rp;
                    double x = up*s*fx + cx;
                    double y = vp*s*fy + cy;
                    if (x >= 0 && x < img_w && y >= 0 && y < img_h)
                    {
                        p[xp] = src.at<unsigned char>(y, x);
                    }
                }
            }
        }
    }


    void CameraBase::pixel2canonical(cv::Point2d& pt) const
    {
        // distortion correction
        undistort(pt);

        // pixel R
        double pan, tilt, roll;
        m_se3.getPanTiltRoll(pan, tilt, roll);
        cv::Matx33d Rp = m_se3.getRotation();

        // canonical R
        evl::SE3 se3;
        se3.setPanTiltRoll(pan, 0, 0);
        cv::Matx33d Rc = se3.getRotation();

        // pixel to canonical
        double u = (pt.x - m_intrinsic.cx) / m_intrinsic.fx;
        double v = (pt.y - m_intrinsic.cy) / m_intrinsic.fy;
        cv::Matx31d ptn(u, v, 1);
        cv::Matx31d ptc = Rc * Rp.t() * ptn;

        pt.x = ptc(0) * m_intrinsic.fx / ptc(2) + m_intrinsic.cx;
        pt.y = ptc(1) * m_intrinsic.fy / ptc(2) + m_intrinsic.cy;
    }

    void CameraBase::pixel2canonical(std::vector<cv::Point2d>& pts) const
    {
        // distortion correction
        undistort(pts);

        // pixel R
        double pan, tilt, roll;
        m_se3.getPanTiltRoll(pan, tilt, roll);
        cv::Matx33d Rp = m_se3.getRotation();

        // canonical R
        evl::SE3 se3;
        se3.setPanTiltRoll(pan, 0, 0);
        cv::Matx33d Rc = se3.getRotation();

        // transform
        cv::Matx33d R = Rc * Rp.t();

        for (int i = 0; i < (int)pts.size(); i++)
        {
            double u = (pts[i].x - m_intrinsic.cx) / m_intrinsic.fx;
            double v = (pts[i].y - m_intrinsic.cy) / m_intrinsic.fy;
            cv::Matx31d ptn(u, v, 1);
            cv::Matx31d ptc = R * ptn;

            pts[i].x = ptc(0) * m_intrinsic.fx / ptc(2) + m_intrinsic.cx;
            pts[i].y = ptc(1) * m_intrinsic.fy / ptc(2) + m_intrinsic.cy;
        }
    }

    void CameraBase::canonical2pixel(cv::Point2d& pt) const
    {
        // pixel R
        double pan, tilt, roll;
        m_se3.getPanTiltRoll(pan, tilt, roll);
        cv::Matx33d Rp = m_se3.getRotation();

        // canonical R
        evl::SE3 se3;
        se3.setPanTiltRoll(pan, 0, 0);
        cv::Matx33d Rc = se3.getRotation();

        // canonical to pixel
        double u = (pt.x - m_intrinsic.cx) / m_intrinsic.fx;
        double v = (pt.y - m_intrinsic.cy) / m_intrinsic.fy;
        cv::Matx31d ptn(u, v, 1);
        cv::Matx31d ptp = Rp * Rc.t() * ptn;

        pt.x = ptp(0) * m_intrinsic.fx / ptp(2) + m_intrinsic.cx;
        pt.y = ptp(1) * m_intrinsic.fy / ptp(2) + m_intrinsic.cy;

        // apply distortion
        distort(pt);
    }

    void CameraBase::canonical2pixel(std::vector<cv::Point2d>& pts) const
    {
        // pixel R
        double pan, tilt, roll;
        m_se3.getPanTiltRoll(pan, tilt, roll);
        cv::Matx33d Rp = m_se3.getRotation();

        // canonical R
        evl::SE3 se3;
        se3.setPanTiltRoll(pan, 0, 0);
        cv::Matx33d Rc = se3.getRotation();

        // transform
        cv::Matx33d R = Rp * Rc.t();

        for (int i = 0; i < (int)pts.size(); i++)
        {
            double u = (pts[i].x - m_intrinsic.cx) / m_intrinsic.fx;
            double v = (pts[i].y - m_intrinsic.cy) / m_intrinsic.fy;
            cv::Matx31d ptn(u, v, 1);
            cv::Matx31d ptp = R * ptn;

            pts[i].x = ptp(0) * m_intrinsic.fx / ptp(2) + m_intrinsic.cx;
            pts[i].y = ptp(1) * m_intrinsic.fy / ptp(2) + m_intrinsic.cy;
        }

        // apply distortion
        distort(pts);
    }

    void CameraBase::distort_normal(cv::Point2d& pt) const
    {
        _distort(pt);
    }

    void CameraBase::distort_normal(std::vector<cv::Point2d>& pts) const
    {
        _distort(pts);
    }

    void CameraBase::undistort_normal(cv::Point2d& pt) const
    {
        _undistort(pt);
    }

    void CameraBase::undistort_normal(std::vector<cv::Point2d>& pts) const
    {
        _undistort(pts);
    }


    void CameraBase::_center_normalize(cv::Point2d& pt) const
    {
        pt.x = pt.x - m_intrinsic.cx;
        pt.y = pt.y - m_intrinsic.cy;
    }

    void CameraBase::_center_normalize(std::vector<cv::Point2d>& pts) const
    {
        for (int i = 0; i < (int)pts.size(); i++)
        {
            pts[i].x = pts[i].x - m_intrinsic.cx;
            pts[i].y = pts[i].y - m_intrinsic.cy;
        }
    }

    void CameraBase::_center_denormalize(cv::Point2d& pt) const
    {
        pt.x = pt.x + m_intrinsic.cx;
        pt.y = pt.y + m_intrinsic.cy;
    }

    void CameraBase::_center_denormalize(std::vector<cv::Point2d>& pts) const
    {
        for (int i = 0; i < (int)pts.size(); i++)
        {
            pts[i].x = pts[i].x + m_intrinsic.cx;
            pts[i].y = pts[i].y + m_intrinsic.cy;
        }
    }

    bool CameraBase::cvtBox2Cylinder(cv::Point3d& p, cv::Size2d& sz, const cv::Rect& box, double offset /*= 0*/) const
    {
        std::vector<cv::Point2d> c_box =
        {
            cv::Point2d(box.x, box.y), cv::Point2d(box.x + box.width, box.y),
            cv::Point2d(box.x, box.y + box.height), cv::Point2d(box.x + box.width, box.y + box.height)
        };
        pixel2canonical(c_box);
        normalize(c_box);

        double x_l = cv::max(c_box[0].x, c_box[2].x);
        double x_r = cv::min(c_box[1].x, c_box[3].x);
        double y_b = (c_box[2].y + c_box[3].y) / 2;
        double y_t = (c_box[0].y + c_box[1].y) / 2;

        double l = (x_r - x_l) / 2;         // The longest radius at both faces
        double x = (x_r + x_l) / 2;

        double R = l * l / y_b * (1 + sqrt(1 + 1 / l / l));
        double Z = R + 1 / y_b;
        double s_b = R / (Z * Z + R * Z);   // The shortest radius at the bottom face
        if (R > 0.3) s_b = 0.3 / (Z * Z + 0.3 * Z);
        double H = (y_b - y_t - s_b) * Z;
        if (H > 1) H = (y_b - y_t) / (1 / Z + s_b);
        if (H < 1) H = (y_b - y_t - s_b - s_b) / (1 / Z - s_b);
        double s_t = s_b * fabs(1 - H);     // The shortest radius at the top face
        if (H < 0)
        {
            H = 0;
            s_t = y_b - y_t - s_b;
        }

        double gx, gy, gz;
        m_se3.getPosition(gx, gy, gz);

        /* world coordinate system of sunglok
        p.z = Z * (gz + offset);
        p.x = x * p.z;
        p.y = offset;
        */
        p.y = Z * (gz + offset);
        p.x = x * p.y;
        p.z = offset;

        sz.width = 2 * R * (gz + offset);
        sz.height = H * (gz + offset);
        return true;
    }

    bool CameraBase::cvtBox2Cylinder(cv::Point2d& foot, cv::Point2d& head, cv::Point3d& p, cv::Size2d& sz, const cv::Rect& box, double offset /*= 0*/) const
    {
        std::vector<cv::Point2d> c_box =
        {
            cv::Point2d(box.x, box.y), cv::Point2d(box.x + box.width, box.y),
            cv::Point2d(box.x, box.y + box.height), cv::Point2d(box.x + box.width, box.y + box.height)
        };
        pixel2canonical(c_box);
        normalize(c_box);

        double x_l = cv::max(c_box[0].x, c_box[2].x);
        double x_r = cv::min(c_box[1].x, c_box[3].x);
        double y_b = (c_box[2].y + c_box[3].y) / 2;
        double y_t = (c_box[0].y + c_box[1].y) / 2;

        double l = (x_r - x_l) / 2;         // The longest radius at both faces
        double x = (x_r + x_l) / 2;

        double R = l * l / y_b * (1 + sqrt(1 + 1 / l / l));
        double Z = R + 1 / y_b;
        double s_b = R / (Z * Z + R * Z);   // The shortest radius at the bottom face
        double H = (y_b - y_t - s_b) * Z;
        if (H > 1) H = (y_b - y_t) / (1 / Z + s_b);
        if (H < 1) H = (y_b - y_t - s_b - s_b) / (1 / Z - s_b);
        double s_t = s_b * fabs(1 - H);     // The shortest radius at the top face
        if (H < 0)
        {
            H = 0;
            s_t = y_b - y_t - s_b;
        }

        foot = cv::Point2d(x, y_b - s_b);
        head = cv::Point2d(x, y_t + s_t);
        denormalize(foot);
        denormalize(head);
        canonical2pixel(foot);
        canonical2pixel(head);

        double gx, gy, gz;
        m_se3.getPosition(gx, gy, gz);

        /* world coordinate system of sunglok
        p.z = Z * (gz + offset);
        p.x = x * p.z;
        p.y = offset;
        */
        p.y = Z * (gz + offset);
        p.x = x * p.y;
        p.z = offset;

        sz.width = 2 * R * (gz + offset);
        sz.height = H * (gz + offset);
        return true;
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    CameraBasic::CameraBasic()
    {
        set_coupled_distortion_model();
    }

    void CameraBasic::_distort(cv::Point2d& pt) const
    {
        double x = pt.x;
        double y = pt.y;
        double r2 = x*x + y*y;
        double radial_term = 1 + m_intrinsic.k1*r2 + m_intrinsic.k2*r2*r2;
        pt.x = radial_term*x + 2 * m_intrinsic.p1*x*y + m_intrinsic.p2*(r2 + 2 * x*x);
        pt.y = radial_term*y + m_intrinsic.p1*(r2 + 2 * y*y) + 2 * m_intrinsic.p2*x*y;
    }

    void CameraBasic::_distort(std::vector<cv::Point2d>& pts) const
    {
        for (int i = 0; i < (int)pts.size(); i++)
        {
            double x = pts[i].x;
            double y = pts[i].y;
            double r2 = x*x + y*y;
            double radial_term = 1 + m_intrinsic.k1*r2 + m_intrinsic.k2*r2*r2;
            pts[i].x = radial_term*x + 2 * m_intrinsic.p1*x*y + m_intrinsic.p2*(r2 + 2 * x*x);
            pts[i].y = radial_term*y + m_intrinsic.p1*(r2 + 2 * y*y) + 2 * m_intrinsic.p2*x*y;
        }
    }

    void CameraBasic::_undistort(cv::Point2d& pt) const
    {
        int max_itr = 100;
        double err_thr = 2 * 0.01 / (m_intrinsic.fx + m_intrinsic.fy);   // 0.01 pixel accuracy

        // Newton method
        double rd = sqrt(pt.x*pt.x + pt.y*pt.y);
        double ru = rd;

        int itr = 0;
        while (itr < max_itr)
        {
            double f = (1 + m_intrinsic.k1*ru*ru + m_intrinsic.k2*ru*ru*ru*ru)*ru - rd;
            double fp = 1 + 3 * m_intrinsic.k1*ru*ru + 5 * m_intrinsic.k2*ru*ru*ru*ru;
            ru = ru - f / fp;
            if (fabs(f) < err_thr) break;
            itr++;
        }

        pt.x = pt.x * ru / rd;
        pt.y = pt.y * ru / rd;
    }

    void CameraBasic::_undistort(std::vector<cv::Point2d>& pts) const
    {
        int max_itr = 100;
        double err_thr = 2 * 0.01 / (m_intrinsic.fx + m_intrinsic.fy);   // 0.01 pixel accuracy

        for (int i = 0; i < (int)pts.size(); i++)
        {
            // Newton method
            double rd = sqrt(pts[i].x*pts[i].x + pts[i].y*pts[i].y);
            double ru = rd;

            int itr = 0;
            while (itr < max_itr)
            {
                double f = (1 + m_intrinsic.k1*ru*ru + m_intrinsic.k2*ru*ru*ru*ru)*ru - rd;
                double fp = 1 + 3 * m_intrinsic.k1*ru*ru + 5 * m_intrinsic.k2*ru*ru*ru*ru;
                ru = ru - f / fp;
                if (fabs(f) < err_thr) break;
                itr++;
            }

            pts[i].x = pts[i].x * ru / rd;
            pts[i].y = pts[i].y * ru / rd;
        }
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    CameraFisheye::CameraFisheye()
    {
        set_coupled_distortion_model();
    }

    void CameraFisheye::_distort(cv::Point2d& pt) const
    {
        // camera matrix
        double m[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        cv::Mat K(3, 3, CV_64FC1, m);

        // distortion coefficients
        double d[] = { m_intrinsic.k1, m_intrinsic.k2, m_intrinsic.k3, m_intrinsic.k4 };
        cv::Mat D(4, 1, CV_64FC1, d);

        std::vector<cv::Point2d> pts1, pts2;
        pts1.push_back(pt);
        cv::fisheye::distortPoints(pts1, pts2, K, D);
        pt = pts2[0];
    }

    void CameraFisheye::_distort(std::vector<cv::Point2d>& pts) const
    {
        // camera matrix
        double m[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        cv::Mat K(3, 3, CV_64FC1, m);

        // distortion coefficients
        double d[] = { m_intrinsic.k1, m_intrinsic.k2, m_intrinsic.k3, m_intrinsic.k4 };
        cv::Mat D(4, 1, CV_64FC1, d);

        std::vector<cv::Point2d> pts_d;
        cv::fisheye::distortPoints(pts, pts_d, K, D);
        pts = pts_d;
    }

    void CameraFisheye::_undistort(cv::Point2d& pt) const
    {
        // camera matrix
        double m[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        cv::Mat K(3, 3, CV_64FC1, m);

        // distortion coefficients
        double d[] = { m_intrinsic.k1, m_intrinsic.k2, m_intrinsic.k3, m_intrinsic.k4 };
        cv::Mat D(4, 1, CV_64FC1, d);

        std::vector<cv::Point2d> pts1, pts2;
        pts1.push_back(pt);
        cv::fisheye::undistortPoints(pts1, pts2, K, D);
        pt = pts2[0];
    }

    void CameraFisheye::_undistort(std::vector<cv::Point2d>& pts) const
    {
        // camera matrix
        double m[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        cv::Mat K(3, 3, CV_64FC1, m);

        // distortion coefficients
        double d[] = { m_intrinsic.k1, m_intrinsic.k2, m_intrinsic.k3, m_intrinsic.k4 };
        cv::Mat D(4, 1, CV_64FC1, d);

        std::vector<cv::Point2d> pts_u;
        cv::fisheye::undistortPoints(pts, pts_u, K, D);
        pts = pts_u;
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    CameraFOV::CameraFOV()
    {
        set_decoupled_distortion_model();
    }

    void CameraFOV::_distort(cv::Point2d& pt) const
    {
        if (m_intrinsic.w <= 0) return;

        double ru = sqrt(pt.x*pt.x + pt.y*pt.y);
        double rd = atan(2 * ru*tan(m_intrinsic.w / 2)) / m_intrinsic.w;

        pt.x = pt.x * rd / ru;
        pt.y = pt.y * rd / ru;
    }

    void CameraFOV::_distort(std::vector<cv::Point2d>& pts) const
    {
        if (m_intrinsic.w <= 0) return;

        for (int i = 0; i < (int)pts.size(); i++)
        {
            double ru = sqrt(pts[i].x*pts[i].x + pts[i].y*pts[i].y);
            double rd = atan(2 * ru*tan(m_intrinsic.w / 2)) / m_intrinsic.w;

            pts[i].x = pts[i].x * rd / ru;
            pts[i].y = pts[i].y * rd / ru;
        }
    }

    void CameraFOV::_undistort(cv::Point2d& pt) const
    {
        if (m_intrinsic.w <= 0) return;

        double rd = sqrt(pt.x*pt.x + pt.y*pt.y);
        double ru = tan(rd*m_intrinsic.w) / (2 * tan(m_intrinsic.w / 2));

        pt.x = pt.x * ru / rd;
        pt.y = pt.y * ru / rd;
    }

    void CameraFOV::_undistort(std::vector<cv::Point2d>& pts) const
    {
        if (m_intrinsic.w <= 0) return;

        for (int i = 0; i < (int)pts.size(); i++)
        {
            double rd = sqrt(pts[i].x*pts[i].x + pts[i].y*pts[i].y);
            double ru = tan(rd*m_intrinsic.w) / (2 * tan(m_intrinsic.w / 2));

            pts[i].x = pts[i].x * ru / rd;
            pts[i].y = pts[i].y * ru / rd;
        }
    }

} // End of 'evl'