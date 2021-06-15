#include "box_tracker.hpp"

using namespace cv;
using namespace std;

#define LOAD_PARAM_VALUE(fn, name_cfg, name_var) \
    if (!(fn)[name_cfg].empty()) (fn)[name_cfg] >> name_var;


BoxTracker::BoxTracker()
{
    m_fps = 15;
}


BoxTracker::~BoxTracker()
{
}


bool BoxTracker::loadParamFile(const cv::String& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    return loadParam(fs.root());
}


bool BoxTracker::loadParam(const cv::FileNode& fn)
{
    // BoxTracker Parameters
    LOAD_PARAM_VALUE(fn, "tracker_velocity_decaying_w", m_param.velocity_decaying_w);
    LOAD_PARAM_VALUE(fn, "tracker_pred_base_length_sec", m_param.pred_base_length_sec);
    LOAD_PARAM_VALUE(fn, "tracker_pred_last_skip_sec", m_param.pred_last_skip_sec);
    LOAD_PARAM_VALUE(fn, "tracker_max_failure_sec", m_param.max_failure_sec);
    LOAD_PARAM_VALUE(fn, "tracker_motion_model", m_param.motion_model);
    LOAD_PARAM_VALUE(fn, "tracker_min_match_score", m_param.min_match_score);
    LOAD_PARAM_VALUE(fn, "tracker_identical_match_score", m_param.identical_match_score);
    LOAD_PARAM_VALUE(fn, "tracker_box_scaleup_w", m_param.box_scaleup_w);
    LOAD_PARAM_VALUE(fn, "tracker_gdist_weight", m_param.gdist_weight);
    LOAD_PARAM_VALUE(fn, "tracker_gdist_sigma", m_param.gdist_sigma);
    LOAD_PARAM_VALUE(fn, "tracker_show_prediction", m_param.show_prediction);
    LOAD_PARAM_VALUE(fn, "tracker_box_width", m_param.box_width);

    LOAD_PARAM_VALUE(fn, "fps", m_fps);
    m_param.smoothing_n = m_fps * 2;
    if (m_param.smoothing_n < 30) m_param.smoothing_n = 30;

    return true;
}


void BoxTracker::update(const std::vector<cv::Rect>& boxes, double image_acquisition_time_sec)
{
    // check empty repository
    if (m_objects.empty())
    {
        for (int i = 0; i < boxes.size(); i++)
            add_new_object(boxes[i], image_acquisition_time_sec);
        return;
    }

    // motion predict
    for (int i = 0; i < (int)m_objects.size(); i++)
    {
        if (m_param.motion_model == 1)
            m_objects[i].rc_pred = predict_advanced(m_objects[i], image_acquisition_time_sec);
        else
            m_objects[i].rc_pred = m_objects[i].rc;
    }

    // compute affinity map
    int M = (int)boxes.size();       // new detections
    int N = (int)m_objects.size();   // current tracking objects

    vector<vector<double>> affinity;
    affinity.resize(M);
    for (int i = 0; i < M; i++)
    {
        affinity[i].resize(N);
        for (int j = 0; j < N; j++)
        {
            affinity[i][j] = calc_affinity(boxes[i], m_objects[j]);
        }
    }

    // compute max affinity for each detection
    vector<double> max_affinity;
    max_affinity.resize(M);
    for (int i = 0; i < M; i++)
    {
        max_affinity[i] = 0;
        for (int j = 0; j < N; j++)
        {
            if (affinity[i][j]>max_affinity[i]) max_affinity[i] = affinity[i][j];
        }
    }

    // find best matches
    vector<int> match;
    match.resize(M);
    for (int i = 0; i < match.size(); i++) match[i] = -1;

    // perform greedy matching
    while (1)
    {
        double maxv = 0;
        int mi, nj;
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                if (affinity[i][j]>maxv)
                {
                    maxv = affinity[i][j];
                    mi = i;
                    nj = j;
                }
        if (maxv < m_param.min_match_score) break;

        match[mi] = nj;
        for (int i = 0; i < M; i++) affinity[i][nj] = -1;
        for (int j = 0; j < N; j++) affinity[mi][j] = -1;
    }

    // update state of tracking list
    for (int j = 0; j < N; j++)
    {
        m_objects[j].detected = false;
    }

    // update matched tracking objects & add new objects
    for (int i = 0; i < M; i++)
    {
        int j = match[i];
        if (j >= 0)
        {
            Point2f p(Point(boxes[i].x + boxes[i].width / 2, boxes[i].y + boxes[i].height / 2));
            Point2f p_prev(Point(m_objects[j].rc.x + m_objects[j].rc.width / 2, m_objects[j].rc.y + m_objects[j].rc.height / 2));
            Point2f velocity = (p - p_prev);
            float alpha = 0.3f;
            m_objects[j].velocity = alpha * velocity + (1 - alpha) * m_objects[j].velocity;

            m_objects[j].rc = boxes[i];
            m_objects[j].traj.push_back(boxes[i]);
            m_objects[j].time_stamp.push_back(image_acquisition_time_sec);
            m_objects[j].detected = true;

            smooth_trajectory(m_objects[j]);
        }
        else
        {
            if (max_affinity[i] < m_param.identical_match_score) add_new_object(boxes[i], image_acquisition_time_sec);
        }
    }

    // remove invalid tracking objects
    int j1 = 0;
    for (int j = 0; j < m_objects.size(); j++)
    {
        int last_i = m_objects[j].time_stamp.size() - 1;
        if ((image_acquisition_time_sec - m_objects[j].time_stamp[last_i]) <= m_param.max_failure_sec)
        {
            if (j != j1) m_objects[j1] = m_objects[j];
            j1++;
        }
    }
    m_objects.resize(j1);
}


void BoxTracker::draw(cv::Mat image)
{
    bool draw_blur = false;
    int unblur_id = 1;
    bool show_trj_pred = false;
    bool show_id = false;

    if (draw_blur && m_blending_mask.empty())
    {
        m_blending_mask = cv::Mat(image.rows, image.cols, CV_8UC3);
        m_blending_mask = cv::Scalar(255, 255, 255);
    }

    for (int j = 0; j < m_objects.size(); j++)
    {
        if( m_param.show_prediction) cv::rectangle(image, m_objects[j].rc_pred, m_objects[j].color, 1);
        if (!m_objects[j].detected) continue;

        if (draw_blur && m_objects[j].id != unblur_id)
        {
            double alpha = 0.2;
            double beta = 1 - alpha;
            double gamma = 0;
            cv::Rect img_rc(0, 0, image.cols, image.rows);
            cv::Rect rc = m_objects[j].rc & img_rc;
            cv::blur(image(rc), image(rc), cv::Size(15, 15));
            addWeighted(m_blending_mask(rc), alpha, image(rc), beta, gamma, image(rc));
        }

        cv::rectangle(image, m_objects[j].rc, m_objects[j].color, m_param.box_width);

        if (show_id)
        {
            string id_str = cv::format("%d", m_objects[j].id);
            putText(image, id_str, m_objects[j].rc.tl(), FONT_HERSHEY_SIMPLEX, 1., Scalar(0, 255, 0), 2);
        }

        if (show_trj_pred)
        {
            for (int k = 1; k < (int)(m_objects[j].trj_pred.size()); k++)
            {
                cv::line(image, m_objects[j].trj_pred[k], m_objects[j].trj_pred[k - 1], m_objects[j].color, 3);
            }
        }
    }
}

Point2f BoxTracker::estimate_decayed_displacement(Point2f velocity_sec_initial, float t_sec, float velocity_decaying_w) const
{
    int n_sec = (int)t_sec;
    float t_remain = t_sec - n_sec;

    Point2f velocity = velocity_sec_initial * velocity_decaying_w;
    Point2f displacement = Point2f(0, 0);
    for (int i = 0; i < n_sec; i++)
    {
        displacement += velocity;
        velocity = velocity * velocity_decaying_w;
    }
    displacement += (t_remain * velocity);

    return displacement;
}

void BoxTracker::smooth_trajectory(box_t& obj)
{
    // extract point trajectory (box center points)
    int n = (int)obj.traj.size();
    obj.smoothed_time_stamp = obj.time_stamp;
    obj.smoothed_pts.resize(n);
    obj.smoothed_rc_w.resize(n);
    obj.smoothed_rc_h.resize(n);
    for (int i = 0; i < n; i++)
    {
        obj.smoothed_pts[i].x = (float)(obj.traj[i].x + obj.traj[i].width / 2);
        obj.smoothed_pts[i].y = (float)(obj.traj[i].y + obj.traj[i].height / 2);
        obj.smoothed_rc_w[i] = (float)(obj.traj[i].width);
        obj.smoothed_rc_h[i] = (float)(obj.traj[i].height);
    }
    if (n <= 2) return; // length 2 trajectory (we don't smooth it)

    // smooth box size in original fps (width, height)
    std::vector<float> rc_w2 = obj.smoothed_rc_w;
    std::vector<float> rc_h2 = obj.smoothed_rc_h;
    for (int k = 0; k < m_param.smoothing_n; k++)
    {
        for (int j = 1; j < n - 1; j++)
        {
            rc_w2[j] = (2 * obj.smoothed_rc_w[j] + obj.smoothed_rc_w[j - 1] + obj.smoothed_rc_w[j + 1]) / 4.0f;
            rc_h2[j] = (2 * obj.smoothed_rc_h[j] + obj.smoothed_rc_h[j - 1] + obj.smoothed_rc_h[j + 1]) / 4.0f;
        }
        rc_w2[0] = (float)(1.5*obj.smoothed_rc_w[0] + obj.smoothed_rc_w[1] + obj.smoothed_rc_w[1] - obj.smoothed_rc_w[2]) / 2.5f;
        rc_w2[n - 1] = (float)(1.5*obj.smoothed_rc_w[n - 1] + obj.smoothed_rc_w[n - 2] + obj.smoothed_rc_w[n - 2] - obj.smoothed_rc_w[n - 3]) / 2.5f;
        rc_h2[0] = (float)(1.5*obj.smoothed_rc_h[0] + obj.smoothed_rc_h[1] + obj.smoothed_rc_h[1] - obj.smoothed_rc_h[2]) / 2.5f;
        rc_h2[n - 1] = (float)(1.5*obj.smoothed_rc_h[n - 1] + obj.smoothed_rc_h[n - 2] + obj.smoothed_rc_h[n - 2] - obj.smoothed_rc_h[n - 3]) / 2.5f;
        obj.smoothed_rc_w = rc_w2;
        obj.smoothed_rc_h = rc_h2;
    }

    // position smoothing in augmented fps (pixel coordinates)
    std::vector<cv::Point2f> pts2 = obj.smoothed_pts;
    for (int k = 0; k < m_param.smoothing_n; k++)
    {
        for (int j = 1; j < n - 1; j++)
        {
            pts2[j] = (2 * obj.smoothed_pts[j] + obj.smoothed_pts[j - 1] + obj.smoothed_pts[j + 1]) / 4.0f;
        }
        pts2[0] = (1.5*obj.smoothed_pts[0] + obj.smoothed_pts[1] + obj.smoothed_pts[1] - obj.smoothed_pts[2]) / 2.5;
        pts2[n - 1] = (1.5*obj.smoothed_pts[n - 1] + obj.smoothed_pts[n - 2] + obj.smoothed_pts[n - 2] - obj.smoothed_pts[n - 3]) / 2.5;
        obj.smoothed_pts = pts2;
    }
}


cv::Rect BoxTracker::predict_advanced(const box_t& obj, double time_t_next) const
{
    int n = (int)obj.traj.size();
    if (n <= 1) return obj.rc;

    // length 2 trajectory (we don't smooth it)
    if (n <= 2)
    {
        double delta_t = (obj.time_stamp[1] - obj.time_stamp[0]);
        if (delta_t <= 0) delta_t = 1;
        Point2f vel_sec = (obj.smoothed_pts[1] - obj.smoothed_pts[0]) / delta_t;
        float t_sec = (float)(time_t_next - obj.time_stamp[1]);
        Point2f pred_pt = obj.smoothed_pts[1] + estimate_decayed_displacement(vel_sec, t_sec, (float)m_param.velocity_decaying_w);

        Rect2f pred_rc;
        pred_rc.width = (1.5f * obj.smoothed_rc_w[1] + obj.smoothed_rc_w[0]) / 2.5f;
        pred_rc.height = (1.5f * obj.smoothed_rc_h[1] + obj.smoothed_rc_h[0]) / 2.5f;
        pred_rc.x = pred_pt.x - pred_rc.width / 2;
        pred_rc.y = pred_pt.y - pred_rc.height / 2;

        return pred_rc;
    }

    // predict box size (simple average of latest box sizes)
    int n_rc_wh = (int)obj.smoothed_rc_w.size();
    Rect2f pred_rc;
    pred_rc.width = obj.smoothed_rc_w[n_rc_wh - 2];
    pred_rc.height = obj.smoothed_rc_h[n_rc_wh - 2];

    // determine frame interval for velocity estimation
    n = (int)obj.smoothed_pts.size();

    double v_step = m_param.pred_base_length_sec;
    double v_skip = m_param.pred_last_skip_sec;

    double frame_et = obj.smoothed_time_stamp[n - 1] - v_skip;
    double frame_st = frame_et - v_step;
    if (frame_st < obj.smoothed_time_stamp[0])
    {
        frame_et = frame_et + (obj.smoothed_time_stamp[0] - frame_st);
        frame_st = obj.smoothed_time_stamp[0];
    }
    if (frame_et > obj.smoothed_time_stamp[n - 1])
    {
        frame_et = obj.smoothed_time_stamp[n - 1];
    }

    int ei = n - 1;
    while (ei > 0 && obj.smoothed_time_stamp[ei] > frame_et) ei--;
    if (ei<(n - 1) && frame_et - obj.smoothed_time_stamp[ei] > obj.smoothed_time_stamp[ei + 1] - frame_et) ei++;
    int si = n - 1;
    while (si > 0 && obj.smoothed_time_stamp[si] > frame_st) si--;
    if (si<ei - 1 && frame_st - obj.smoothed_time_stamp[si] > obj.smoothed_time_stamp[si + 1] - frame_st) si++;
    if (ei == si)
    {
        ei++;
        if (ei > n - 1)
        {
            ei = n - 1;
            si = n - 2;
        }
    }

    // predict next point
    double delta_t = obj.smoothed_time_stamp[ei] - obj.smoothed_time_stamp[si];
    if (delta_t <= 0) delta_t = 1;
    Point2f vel_sec = (obj.smoothed_pts[ei] - obj.smoothed_pts[si]) / delta_t;
    float t_sec = (float)(time_t_next - obj.smoothed_time_stamp[ei]);
    Point2f pred_pt = obj.smoothed_pts[ei] + estimate_decayed_displacement(vel_sec, t_sec, (float)m_param.velocity_decaying_w);
    pred_rc.x = pred_pt.x - pred_rc.width / 2;
    pred_rc.y = pred_pt.y - pred_rc.height / 2;

    return pred_rc;
}


double BoxTracker::calc_affinity(cv::Rect rc, box_t obj) const
{
    cv::Rect rc1 = rc;
    cv::Rect rc2 = obj.rc_pred;

    // scale up
    if (m_param.box_scaleup_w > 1)
    {
        double s = m_param.box_scaleup_w;
        rc1 = Rect((int)(rc1.x + rc1.width / 2 - rc1.width*s / 2), (int)(rc1.y + rc1.height / 2 - rc1.height*s / 2), (int)(rc1.width*s), (int)(rc1.height*s));
        rc2 = Rect((int)(rc2.x + rc2.width / 2 - rc2.width*s / 2), (int)(rc2.y + rc2.height / 2 - rc2.height*s / 2), (int)(rc2.width*s), (int)(rc2.height*s));
    }

    // overlap affinity
    double overlap_affinity = calc_overlap(rc1, rc2);

    // distance affinity
    Point p1(rc1.x + rc1.width / 2, rc1.y + rc1.height / 2);
    Point p2(rc2.x + rc2.width / 2, rc2.y + rc2.height / 2);
    double gdist_affinity = exp(-((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y))/(m_param.gdist_sigma*m_param.gdist_sigma));

    return overlap_affinity*(1-m_param.gdist_weight) + gdist_affinity*m_param.gdist_weight;
}


double BoxTracker::calc_overlap(cv::Rect rc1, cv::Rect rc2) const
{
    cv::Rect rc_intersect = rc1 & rc2;
    double area_intersect = rc_intersect.area();
    double area_union = rc1.area() + rc2.area() - area_intersect;
    return area_intersect / area_union;
}

void BoxTracker::add_new_object(cv::Rect rc, double time_t, bool detected)
{
    box_t box;
    box.id = m_idGenerator.getID();
    box.color = m_colorGenerator.getColor();
    box.rc = rc;
    box.rc_pred = rc;
    box.traj.push_back(rc);
    box.time_stamp.push_back(time_t);
    box.detected = detected;
    box.velocity = Point2f(0, 0);

    Point2f smoothed_pt;
    smoothed_pt.x = rc.x + rc.width / 2;
    smoothed_pt.y = rc.y + rc.height / 2;
    box.smoothed_pts.push_back(smoothed_pt);
    box.smoothed_rc_w.push_back((float)rc.width);
    box.smoothed_rc_h.push_back((float)rc.height);
    box.smoothed_time_stamp.push_back(time_t);

    m_objects.push_back(box);
}
