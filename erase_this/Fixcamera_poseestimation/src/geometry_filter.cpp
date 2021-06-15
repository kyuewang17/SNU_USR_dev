#include "geometry_filter.hpp"


#define LOAD_PARAM_VALUE(fn, name_cfg, name_var) \
    if (!(fn)[name_cfg].empty()) (fn)[name_cfg] >> name_var;


GeometryFilter::~GeometryFilter()
{
}


void GeometryFilter::setCamera(evl::CameraFOV* camera)
{
    m_camera = camera;
}


void GeometryFilter::setImageSize(int image_width, int image_height)
{
    m_img_w = image_width;
    m_img_h = image_height;
}


bool GeometryFilter::loadParamFile(const cv::String& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    return loadParam(fs.root());
}


bool GeometryFilter::loadParam(const cv::FileNode& fn)
{
    // Common Parameters
    int image_width = -1;
    int image_height = -1;
    LOAD_PARAM_VALUE(fn, "img_w", image_width);
    LOAD_PARAM_VALUE(fn, "img_h", image_height);
    if (image_width>0 && image_height>0) setImageSize(image_width, image_height);

    // Geometric Filter Parameters
    GeometryFilterParam geoParam = getParam();
    LOAD_PARAM_VALUE(fn, "geofilter_img_boundary_margin", geoParam.img_boundary_margin);
    LOAD_PARAM_VALUE(fn, "geofilter_restore_occlusion", geoParam.restore_occlusion);
    LOAD_PARAM_VALUE(fn, "geofilter_restore_ratio", geoParam.restore_ratio);
    LOAD_PARAM_VALUE(fn, "geofilter_remove_far_object", geoParam.remove_far_object);
    LOAD_PARAM_VALUE(fn, "geofilter_far_object_size", geoParam.far_object_size);
    LOAD_PARAM_VALUE(fn, "geofilter_remove_duplication", geoParam.remove_duplication);
    LOAD_PARAM_VALUE(fn, "geofilter_duplication_overlap", geoParam.duplication_overlap);
    LOAD_PARAM_VALUE(fn, "geofilter_base_object_size", geoParam.base_object_size);
    LOAD_PARAM_VALUE(fn, "geofilter_min_aspect_ratio", geoParam.min_aspect_ratio);
    LOAD_PARAM_VALUE(fn, "geofilter_max_aspect_ratio", geoParam.max_aspect_ratio);
    LOAD_PARAM_VALUE(fn, "geofilter_min_height_multiplier", geoParam.min_height_multiplier);
    LOAD_PARAM_VALUE(fn, "geofilter_max_height_multiplier", geoParam.max_height_multiplier);
    LOAD_PARAM_VALUE(fn, "geofilter_min_size_multiplier", geoParam.min_size_multiplier);
    LOAD_PARAM_VALUE(fn, "geofilter_max_size_multiplier", geoParam.max_size_multiplier);
    LOAD_PARAM_VALUE(fn, "verbose_geofilter", geoParam.verbose);
    setParam(geoParam);

    return true;
}


void GeometryFilter::correct(std::vector<cv::Rect>& boxes)
{
    if (m_camera == NULL) return;

    if (m_param.remove_duplication) remove_duplication(boxes);
    if (m_param.restore_occlusion) height_compensation(boxes);
    if (m_param.remove_far_object) remove_far_objects(boxes);
    remove_false(boxes);
}


void GeometryFilter::remove_false(std::vector<cv::Rect>& boxes)
{
    std::vector<cv::Rect> tmp;
    for (int i = 0; i < boxes.size(); i++)
    {
        cv::Point2d top_est;
        int ix = boxes[i].x + boxes[i].width / 2;
        int iy = boxes[i].y + boxes[i].height;
        bool ok = estimate_top(ix, iy, top_est);
        double h_est = iy - top_est.y;
        double wdelta_est = fabs(ix - top_est.x)/2;

        double aspect = (double)(boxes[i].width) / boxes[i].height;
        bool aspect_ok = (aspect >= m_param.min_aspect_ratio && aspect <= m_param.max_aspect_ratio);

        double hr= boxes[i].height / h_est;
        bool hr_ok = (hr >= m_param.min_height_multiplier && hr <= m_param.max_height_multiplier);

        double size_est = h_est * (h_est / 2 + wdelta_est);
        double sr = boxes[i].area() / size_est;
        bool sr_ok = (sr >= m_param.min_size_multiplier && sr <= m_param.max_size_multiplier);

        if (ok && aspect_ok && hr_ok && sr_ok)
        {
            tmp.push_back(boxes[i]);
        }
        if (m_param.verbose>0)
        {
            if (!sr_ok) printf("false: sr = %lf\n", sr);
            if (!hr_ok) printf("false: hr = %lf\n", hr);
            if (!aspect_ok) printf("false: aspect = %lf\n", aspect);
        }
    }
    boxes = tmp;
}

void GeometryFilter::height_compensation(std::vector<cv::Rect>& boxes)
{
    // 가려짐으로 인한 부분적 잘림을 복원
    for (int i = 0; i < boxes.size(); i++)
    {
        // 이미지 상단에서 잘린 경우 제외
        if (boxes[i].y <= m_param.img_boundary_margin) continue;

        cv::Point2d bottom_est;
        bool ok = estimate_bottom(boxes[i].x, boxes[i].y, bottom_est);

        // 이미 검출된 높이가 커서 복원할 필요가 없는 경우 제외
        int bottom_y = boxes[i].y + boxes[i].height;
        double h_est = bottom_est.y - boxes[i].y;
        if (!ok || boxes[i].height >= h_est* m_param.restore_ratio) continue;

        // find intersecting box (가려짐 검출)
        double max_overlap = 0;
        int overlap_box_index = -1;
        for (int j = 0; j < boxes.size(); j++)
        {
            if (j == i) continue;
            if (boxes[j].y <= boxes[i].y) continue;
            if (boxes[j].y + boxes[j].height < bottom_y) continue;
            double overlap = calc_overlap(boxes[j], boxes[i]);
            if (overlap > max_overlap)
            {
                max_overlap = overlap;
                overlap_box_index = j;
            }
        }

        double aspect = boxes[i].width / h_est;
        if (max_overlap > 0 || (aspect < 0.7 && bottom_y >= m_img_h - m_param.img_boundary_margin))
        {
            boxes[i].height = (int)(h_est * m_param.restore_ratio);
        }
    }
}

void GeometryFilter::remove_duplication(std::vector<cv::Rect>& boxes)
{
    std::vector<cv::Rect> tmp;
    for (int i = 0; i < boxes.size(); i++)
    {
        bool duplication = false;
        double overlap1;
        for (int j = 0; j < boxes.size(); j++)
        {
            if (j == i) continue;
            if (boxes[j].y > boxes[i].y) continue;
            if (boxes[j].y + boxes[j].height < boxes[i].y + boxes[i].height) continue;
            calc_overlap(boxes[i], boxes[j], &overlap1);
            if (overlap1 > m_param.duplication_overlap)
            {
                duplication = true;
                break;
            }
        }

        if (!duplication)
        {
            tmp.push_back(boxes[i]);
        }
        if (m_param.verbose>0 && duplication)
        {
            printf("duplication: r=%lf, x=%d, y=%d, w=%d, h=%d\n", overlap1, boxes[i].x, boxes[i].y, boxes[i].width, boxes[i].height);
        }
    }
    boxes = tmp;
}


void GeometryFilter::remove_far_objects(std::vector<cv::Rect>& boxes)
{
    std::vector<cv::Rect> tmp;
    for (int i = 0; i < boxes.size(); i++)
    {
        int bottom_x = boxes[i].x + boxes[i].width / 2;
        int bottom_y = boxes[i].y + boxes[i].height;

        cv::Point2d top;
        bool ok = estimate_top(bottom_x, bottom_y, top);
        if (ok && (bottom_y - top.y)>m_param.far_object_size)
        {
            tmp.push_back(boxes[i]);
        }
        else if (m_param.verbose>0)
        {
            printf("far object: y=%d, est_h=%.0lf, img_h=%d\n", bottom_y, bottom_y - top.y, boxes[i].height);
        }

    }
    boxes = tmp;
}


bool GeometryFilter::estimate_top(double ix, double iy, cv::Point2d& top)
{
    cv::Point2d pt1(ix, iy), pt1_ground;
    bool ok = m_camera->unproject_ground(pt1, pt1_ground);
    if (!ok)
    {
        top = pt1;
        return false;
    }

    cv::Point3d pt2_world(pt1_ground.x, pt1_ground.y, m_param.base_object_size);
    ok = m_camera->project(pt2_world, top);
    return ok;
}

bool GeometryFilter::estimate_bottom(double ix, double iy, cv::Point2d& bottom)
{
    cv::Point2d pt(ix, iy);
    m_camera->undistort(pt);

    evl::CameraParam param = m_camera->get_intrinsic_parameters();
    double nx = (pt.x - param.cx) / param.fx;
    double ny = (pt.y - param.cy) / param.fy;

    cv::Matx33d R;
    cv::Matx31d t;
    m_camera->get_extrinsic().getRt(R, t);

    cv::Matx31d Nc(nx, ny, 1);
    cv::Matx31d RN = R.t()*Nc;
    cv::Matx31d Rt = R.t()*t;
    double depth = (m_param.base_object_size + Rt(2)) / RN(2);       // Pw = Rinv*(Pc - t), Pc = depth*Nc
    cv::Matx31d Pw = depth*RN - Rt;
    cv::Point3d pt_bottom_world(Pw(0), Pw(1), 0);
    bool ok = m_camera->project(pt_bottom_world, bottom);

    if (ok && bottom.y > iy) return true;
    else return false;
}

bool GeometryFilter::is_boundary(cv::Rect rc)
{
    int d = m_param.img_boundary_margin;
    if (rc.x<d || rc.y<d || rc.x + rc.width>=m_img_w - d || rc.y + rc.height>=m_img_h - d) return true;
    else return false;
}

double GeometryFilter::calc_overlap(cv::Rect rc1, cv::Rect rc2, double* overlap1, double* overlap2) const
{
    cv::Rect rc_intersect = rc1 & rc2;
    double area_intersect = rc_intersect.area();
    double area_union = rc1.area() + rc2.area() - area_intersect;
    if (overlap1) *overlap1 = area_intersect / rc1.area();
    if (overlap2) *overlap2 = area_intersect / rc2.area();
    return area_intersect / area_union;
}
