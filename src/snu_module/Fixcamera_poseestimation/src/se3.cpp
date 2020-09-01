#include "se3.hpp"
#include "opencv/cxcore.hpp"
#include "opencv2/calib3d.hpp"
#include <math.h>

#define RAD2DEG(x)         ((x) * 180.0 / CV_PI)
#define DEG2RAD(x)         ((x) * CV_PI / 180.0)

using namespace cv;

namespace evl
{

    ////////////////////////////////////////////////////////////////////////////////////////////
    // constructors
    SE3::SE3(void)
    {
        setIdentity();
    }

    SE3::SE3(const Matx33d& R, const Matx31d& t)
    : m_R(R), m_t(t)
    {
    }

    SE3::SE3(const Matx34d& Rt)
    {
        setRt(Rt);
    }

    SE3::SE3(const Mat& R, const Mat& t)
    {
        setRt(R,t);
    }

    SE3::SE3(const Mat& Rt)
    {
        setRt(Rt);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    // operations
    void SE3::inv()
    {
        m_R = m_R.t();
        m_t = -m_R*m_t;
    }

    void SE3::setIdentity()
    {
        m_R = Matx33d::eye();
        m_t = Matx31d::zeros();
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    // 회전변환(R), 평행이동(t)
    const Matx34d SE3::getRt() const
    {
        Matx34d Rt;

        Rt(0,0) = m_R(0,0);
        Rt(0,1) = m_R(0,1);
        Rt(0,2) = m_R(0,2);
        Rt(1,0) = m_R(1,0);
        Rt(1,1) = m_R(1,1);
        Rt(1,2) = m_R(1,2);
        Rt(2,0) = m_R(2,0);
        Rt(2,1) = m_R(2,1);
        Rt(2,2) = m_R(2,2);

        Rt(0,3) = m_t(0);
        Rt(1,3) = m_t(1);
        Rt(2,3) = m_t(2);

        return Rt;
    }

    void SE3::getRt(cv::Matx34d& Rt) const
    {
        Rt = getRt();
    }

    void SE3::setRt(const Matx34d& Rt)
    {
        m_R(0,0) = Rt(0,0);
        m_R(0,1) = Rt(0,1);
        m_R(0,2) = Rt(0,2);
        m_R(1,0) = Rt(1,0);
        m_R(1,1) = Rt(1,1);
        m_R(1,2) = Rt(1,2);
        m_R(2,0) = Rt(2,0);
        m_R(2,1) = Rt(2,1);
        m_R(2,2) = Rt(2,2);

        m_t(0) = Rt(0,3);
        m_t(1) = Rt(1,3);
        m_t(2) = Rt(2,3);
    }

    void SE3::setRt(const Mat& Rt)
    {
        CV_Assert((Rt.type()==CV_64FC1 || Rt.type()==CV_32FC1) && Rt.rows==3 && Rt.cols==4);

        if(Rt.type()==CV_64FC1)
        {
            m_R(0,0) = Rt.at<double>(0,0);
            m_R(0,1) = Rt.at<double>(0,1);
            m_R(0,2) = Rt.at<double>(0,2);
            m_R(1,0) = Rt.at<double>(1,0);
            m_R(1,1) = Rt.at<double>(1,1);
            m_R(1,2) = Rt.at<double>(1,2);
            m_R(2,0) = Rt.at<double>(2,0);
            m_R(2,1) = Rt.at<double>(2,1);
            m_R(2,2) = Rt.at<double>(2,2);

            m_t(0) = Rt.at<double>(0,3);
            m_t(1) = Rt.at<double>(1,3);
            m_t(2) = Rt.at<double>(2,3);
        }
        else
        {
            m_R(0,0) = Rt.at<float>(0,0);
            m_R(0,1) = Rt.at<float>(0,1);
            m_R(0,2) = Rt.at<float>(0,2);
            m_R(1,0) = Rt.at<float>(1,0);
            m_R(1,1) = Rt.at<float>(1,1);
            m_R(1,2) = Rt.at<float>(1,2);
            m_R(2,0) = Rt.at<float>(2,0);
            m_R(2,1) = Rt.at<float>(2,1);
            m_R(2,2) = Rt.at<float>(2,2);

            m_t(0) = Rt.at<float>(0,3);
            m_t(1) = Rt.at<float>(1,3);
            m_t(2) = Rt.at<float>(2,3);
        }
    }

    void SE3::getRt(Matx33d& R, Matx31d& t) const
    {
        R = m_R;
        t = m_t;
    }

    void SE3::setRt(const Matx33d& R, const Matx31d& t)
    {
        m_R = R;
        m_t = t;
    }

    void SE3::setRt(const Mat& R, const Mat& t)
    {
        CV_Assert(R.rows==3 && R.cols==3 && t.rows==3 && t.cols==1);

        R.convertTo(m_R, CV_64FC1);
        t.convertTo(m_t, CV_64FC1);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    // 회전변환(R) & 3D 방향(orientation)
    const Matx33d& SE3::getRotation() const
    {
        return m_R;
    }

    void SE3::setRotation(const Matx33d& R)
    {
        m_R = R;
    }

    void SE3::setRotation(const Mat& R)
    {
        CV_Assert(R.rows==3 && R.cols==3);
        R.convertTo(m_R, CV_64FC1);
    }

    void SE3::setRotationPinned(const Matx33d& R)
    {
        m_t = R*m_R.t()*m_t;
        m_R = R;
    }

    void SE3::setRotationPinned(const Mat& R)
    {
        CV_Assert(R.rows==3 && R.cols==3);

        Matx33d Rd;
        R.convertTo(Rd, CV_64FC1);

        setRotationPinned(Rd);
    }

    void SE3::getEulerAngles(double& pitch, double& roll, double& yaw) const
    {
        int s = 1;
        if (m_R(0,0) < 0) s = -1;
        double norm = sqrt(m_R(2,1) * m_R(2,1) + m_R(2,2) * m_R(2,2));

        pitch = atan2(s * m_R(2,1), s * m_R(2,2));
        roll = atan2(-m_R(2,0), s * norm);
        yaw = atan2(s * m_R(1,0), s * m_R(0,0));
    }

    void SE3::setEulerAngles(double pitch, double roll, double yaw)
    {
        m_R = rotationFromEulerAngles(pitch, roll, yaw);
    }

    void SE3::setEulerAnglesPinned(double pitch, double roll, double yaw)
    {
        Matx33d R = rotationFromEulerAngles(pitch, roll, yaw);
        setRotationPinned(R);
    }

    void SE3::getEulerAnglesGraphics(double& pitch, double& roll, double& yaw) const
    {
        Matx33d V2G(1,  0,  0,
                    0, -1,  0,
                    0,  0, -1);

        Matx33d GR = V2G * m_R;			// convert to graphics camera

        // check degenerate cases (roll == +-CV_PI/2)
        if(GR(0,0)==0 && GR(0,1)==0 || GR(1,2)==0 && GR(2,2)==0)
        {
            int sign = (GR(0,2)>0) ? 1: -1;
            roll = -sign * CV_PI/2;
            pitch = 0;
            yaw = -sign * atan2(sign * GR(1,0), GR(1,1));
        }
        else
        {
            yaw = -atan2(-GR(0,1),GR(0,0));
            roll = -asin(GR(0,2));
            pitch = -atan2(-GR(1,2),GR(2,2));
        }
    }

    void SE3::setEulerAnglesGraphics(double pitch, double roll, double yaw)
    {
        m_R = rotationFromEulerAnglesGraphics(pitch, roll, yaw);
    }

    void SE3::setEulerAnglesGraphicsPinned(double pitch, double roll, double yaw)
    {
        Matx33d R = rotationFromEulerAnglesGraphics(pitch, roll, yaw);
        setRotationPinned(R);
    }

    void SE3::getRodrigues(Mat& rvec) const
    {
        Rodrigues(m_R, rvec);
    }

    void SE3::setRodrigues(const Mat& rvec)
    {
        m_R = rotationFromRodrigues(rvec);
    }

    void SE3::setRodriguesPinned(const Mat& rvec)
    {
        Matx33d R = rotationFromRodrigues(rvec);
        setRotationPinned(R);
    }

    void SE3::getRodrigues(double& vx, double& vy, double& vz, double& theta_radian) const
    {
        Mat rvec;
        getRodrigues(rvec);

        theta_radian = norm(rvec);
        vx = rvec.at<double>(0) / theta_radian;
        vy = rvec.at<double>(1) / theta_radian;
        vz = rvec.at<double>(2) / theta_radian;
    }

    void SE3::setRodrigues(double vx, double vy, double vz, double theta_radian)
    {
        m_R = rotationFromRodrigues(vx, vy, vz, theta_radian);
    }

    void SE3::setRodriguesPinned(double vx, double vy, double vz, double theta_radian)
    {
        Matx33d R = rotationFromRodrigues(vx, vy, vz, theta_radian);
        setRotationPinned(R);
    }

    void SE3::getQuaternion(double& w, double& x, double& y, double& z) const
    {
        double t = m_R(0,0) + m_R(1,1) + m_R(2,2);
        double r = sqrt(1+t);

        w = 0.5*r;
        x = 0.5*sqrt(1+m_R(0,0)-m_R(1,1)-m_R(2,2));
        y = 0.5*sqrt(1-m_R(0,0)+m_R(1,1)-m_R(2,2));
        z = 0.5*sqrt(1-m_R(0,0)-m_R(1,1)+m_R(2,2));

        if(m_R(2,1)<m_R(1,2)) x = -x;
        if(m_R(0,2)<m_R(2,0)) y = -y;
        if(m_R(1,0)<m_R(0,1)) z = -z;
    }

    void SE3::setQuaternion(double w, double x, double y, double z)
    {
        m_R = rotationFromQuaternion(w, x, y, z);
    }

    void SE3::setQuaternionPinned(double w, double x, double y, double z)
    {
        Matx33d R = rotationFromQuaternion(w, x, y, z);
        setRotationPinned(R);
    }

    void SE3::getPanTiltRoll(double& pan, double& tilt, double& roll) const
    {
        Matx33d R_inv = m_R.t();

        Matx31d zc(0,0,1);		// camera coordinate of camera z axis
        Matx31d zw = R_inv*zc;	// world coordinate of camera z axis

        pan = atan2(zw(1),zw(0));
        tilt = atan2(zw(2), sqrt(zw(0)*zw(0)+zw(1)*zw(1)));

        Matx31d xc(1,0,0);		// camera coordinate of camera x axis
        Matx31d xw = R_inv*xc;	// world coordinate of camera x axis
        Matx31d xp(cos(pan-CV_PI/2), sin(pan-CV_PI/2), 0);	// world coordinate of camera x axis without roll

        double v = xw.dot(xp);
        if(v>1) v = 1;
        if(v<-1) v = -1;
        roll = acos(v);
        if(xw(2)<0) roll = -roll;
    }

    void SE3::setPanTiltRoll(double pan, double tilt, double roll)
    {
        m_R = rotationFromPanTiltRoll(pan, tilt, roll);
    }

    void SE3::setPanTiltRollPinned(double pan, double tilt, double roll)
    {
        Matx33d R = rotationFromPanTiltRoll(pan, tilt, roll);
        setRotationPinned(R);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    // 평행이동(t) & 3D 위치(position)
    const Matx31d& SE3::getTranslation() const
    {
        return m_t;
    }

    void SE3::setTranslation(const Matx31d& t)
    {
        m_t = t;
    }

    void SE3::setTranslation(const Mat& t)
    {
        CV_Assert(t.rows==3 && t.cols==1);
        t.convertTo(m_t, CV_64FC1);
    }

    void SE3::getTranslation(double& tx, double& ty, double& tz) const
    {
        tx = m_t(0);
        ty = m_t(1);
        tz = m_t(2);
    }

    void SE3::setTranslation(double tx, double ty, double tz)
    {
        m_t(0) = tx;
        m_t(1) = ty;
        m_t(2) = tz;
    }

    void SE3::getPosition(double& x, double& y, double& z) const
    {
        Matx31d p = -m_R.t()*m_t;

        x = p(0);
        y = p(1);
        z = p(2);
    }

    void SE3::setPosition(double x, double y, double z)
    {
        Matx31d p(x,y,z);

        m_t = -m_R*p;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    // 3D 자세(pose)
    void SE3::getPose(double& x, double& y, double& z, double& pitch, double& roll, double& yaw) const
    {
        getEulerAngles(pitch, roll, yaw);
        getPosition(x, y, z);
    }

    void SE3::setPose(double x, double y, double z, double pitch, double roll, double yaw)
    {
        setEulerAnglesPinned(pitch, roll, yaw);
        setPosition(x, y, z);
    }

    void SE3::getPoseGraphics(double& x, double& y, double& z, double& pitch, double& roll, double& yaw) const
    {
        getEulerAnglesGraphics(pitch, roll, yaw);
        getPosition(x, y, z);
    }

    void SE3::setPoseGraphics(double x, double y, double z, double pitch, double roll, double yaw)
    {
        setEulerAnglesGraphicsPinned(pitch, roll, yaw);
        setPosition(x, y, z);
    }

    void SE3::getPosePanTiltRoll(double& x, double& y, double& z, double& pan, double& tilt, double& roll) const
    {
        getPanTiltRoll(pan, tilt, roll);
        getPosition(x, y, z);
    }

    void SE3::setPosePanTiltRoll(double x, double y, double z, double pan, double tilt, double roll)
    {
        setPanTiltRollPinned(pan, tilt, roll);
        setPosition(x, y, z);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    // utilities
    void SE3::display() const
    {
        printf("[R|t] = \n");
        for(int r=0; r<3; r++)
        {
            printf("%lf %lf %lf  |  %lf\n", m_R(r,0), m_R(r,1), m_R(r,2), m_t(r));
        }

        Matx31d p = -m_R.t()*m_t;
        printf("x=%lf y=%lf z=%lf\n", p(0), p(1), p(2));
    }

    void SE3::display(const Mat& mat)
    {
        if(mat.channels()>1) return;

        int nr = mat.rows;
        int nc = mat.cols;

        Mat m;
        mat.convertTo(m, CV_64FC1);

        for(int r=0; r<nr; r++)
        {
            for(int c=0; c<nc; c++)
            {
                printf("%lf ", m.at<double>(r,c));
            }
            printf("\n");
        }
    }

    Matx33d SE3::rotationMatX(double rad)
    {
        double c = cos(rad);
        double s = sin(rad);

        Matx33d R(1, 0, 0,
                  0, c,-s,
                  0, s, c);

        return R;
    }

    Matx33d SE3::rotationMatY(double rad)
    {
        double c = cos(rad);
        double s = sin(rad);

        Matx33d R(c, 0, s,
                  0, 1, 0,
                 -s, 0, c);

        return R;
    }

    Matx33d SE3::rotationMatZ(double rad)
    {
        double c = cos(rad);
        double s = sin(rad);

        Matx33d R(c,-s, 0,
                  s, c, 0,
                  0, 0, 1);

        return R;
    }

    Matx33d SE3::rotationFromEulerAngles(double pitch, double roll, double yaw)
    {
        double c,s;

        c = cos(pitch), s = sin(pitch);
        Matx33d Rx( 1,  0,  0,
                    0,  c, -s,
                    0,  s,  c);

        c = cos(roll), s = sin(roll);
        Matx33d Ry( c,  0,  s,
                    0,  1,  0,
                   -s,  0,  c);

        c = cos(yaw), s = sin(yaw);
        Matx33d Rz( c, -s,  0,
                    s,  c,  0,
                    0,  0,  1);

        Matx33d R = Rz * Ry * Rx;

        return R;
    }

    Matx33d SE3::rotationFromEulerAnglesGraphics(double pitch, double roll, double yaw)
    {
        double sx = sin(-pitch);
        double cx = cos(-pitch);
        double sy = sin(-roll);
        double cy = cos(-roll);
        double sz = sin(-yaw);
        double cz = cos(-yaw);

        Matx33d R;
        R(0,0) = cy*cz;				R(0,1) = -cy*sz;			R(0,2) = sy;
        R(1,0) = -sx*sy*cz-cx*sz;	R(1,1) = sx*sy*sz-cx*cz;	R(1,2) = sx*cy;
        R(2,0) = cx*sy*cz-sx*sz;	R(2,1) = -cx*sy*sz-sx*cz;	R(2,2) = -cx*cy;

        return R;
    }

    Matx33d SE3::rotationFromRodrigues(const cv::Mat& rvec)
    {
        Matx33d R;
        Rodrigues(rvec, R);

        return R;
    }

    Matx33d SE3::rotationFromRodrigues(double vx, double vy, double vz, double theta_radian)
    {
        double s = sqrt(vx*vx + vy*vy + vz*vz);
        double rv_data[3] = {vx*theta_radian/s, vy*theta_radian/s, vz*theta_radian/s};
        Mat rvec(3, 1, CV_64FC1, rv_data);
        Matx33d R = rotationFromRodrigues(rvec);

        return R;
    }

    Matx33d SE3::rotationFromQuaternion(double w, double x, double y, double z)
    {
        double n = w * w + x * x + y * y + z * z;
        double s = (n==0) ? 0 : 2/n;
        double wx = s * w * x;
        double wy = s * w * y;
        double wz = s * w * z;
        double xx = s * x * x;
        double xy = s * x * y;
        double xz = s * x * z;
        double yy = s * y * y;
        double yz = s * y * z;
        double zz = s * z * z;

        Matx33d R;
        R(0,0) = 1 - (yy + zz); R(0,1) = xy - wz;       R(0,2) = xz + wy;
        R(1,0) = xy + wz;       R(1,1) = 1 - (xx + zz); R(1,2) = yz - wx;
        R(2,0) = xz - wy;       R(2,1) = yz + wx;       R(2,2) = 1 - (xx + yy);

        return R;
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

} // End of 'evl'