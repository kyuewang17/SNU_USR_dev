#include "id_color_generator.hpp"

ColorGenerator::ColorGenerator()
{
	initMediumSet();
}

ColorGenerator::~ColorGenerator()
{
}

void ColorGenerator::initTinySet()
{
	int k=0;

	m_cr[k++] = cv::Scalar(255,255,255);

	m_cr[k++] = cv::Scalar(255,0,0);
	m_cr[k++] = cv::Scalar(0,255,0);
	m_cr[k++] = cv::Scalar(0,0,255);

	m_cr[k++] = cv::Scalar(255,255,0);
	m_cr[k++] = cv::Scalar(255,0,255);
	m_cr[k++] = cv::Scalar(0,255,255);


	m_index = 0;
	m_count = k;
}

void ColorGenerator::initMediumSet()
{
	int k=0;

    m_cr[k++] = cv::Scalar(255,0,255);
    m_cr[k++] = cv::Scalar(255, 255, 0);
    m_cr[k++] = cv::Scalar(0, 255, 255);

    m_cr[k++] = cv::Scalar(128, 255, 0);
    m_cr[k++] = cv::Scalar(0, 128, 255);
    m_cr[k++] = cv::Scalar(255, 0, 128);

    m_cr[k++] = cv::Scalar(0, 255, 128);
    m_cr[k++] = cv::Scalar(255, 128, 0);
    m_cr[k++] = cv::Scalar(128, 0, 255);

    m_cr[k++] = cv::Scalar(128, 0, 0);
    m_cr[k++] = cv::Scalar(0, 128, 0);
    m_cr[k++] = cv::Scalar(0, 0, 128);

    m_cr[k++] = cv::Scalar(255, 255, 255);

    m_cr[k++] = cv::Scalar(128, 128, 0);
    m_cr[k++] = cv::Scalar(128, 0, 128);
    m_cr[k++] = cv::Scalar(0, 128, 128);

    m_cr[k++] = cv::Scalar(255, 0, 0);
    m_cr[k++] = cv::Scalar(0, 255, 0);

    m_cr[k++] = cv::Scalar(255, 128, 128);
    m_cr[k++] = cv::Scalar(128, 128, 255);
    m_cr[k++] = cv::Scalar(128, 255, 128);

    m_cr[k++] = cv::Scalar(0, 0, 255);

    m_cr[k++] = cv::Scalar(192, 0, 64);
    m_cr[k++] = cv::Scalar(0, 64, 192);
    m_cr[k++] = cv::Scalar(64, 192, 0);

    m_cr[k++] = cv::Scalar(255, 128, 255);
    m_cr[k++] = cv::Scalar(128, 255, 255);
    m_cr[k++] = cv::Scalar(255, 255, 128);

    m_cr[k++] = cv::Scalar(64, 0, 192);
    m_cr[k++] = cv::Scalar(0, 192, 64);
    m_cr[k++] = cv::Scalar(192, 64, 0);

    m_cr[k++] = cv::Scalar(0,0,0);

	m_index = 0;
	m_count = k;
}


void ColorGenerator::initLargeSet()
{
	int c[5] = {255, 192, 128, 64, 0};

	m_index = 0;
	m_count = 5*5*5;

	int i=0;
	for(int r=0; r<5; r++)
	{
		for(int g=0; g<5; g++)
		{
			for(int b=0; b<5; b++)
			{
				i = (i + 17) % m_count;
				m_cr[i] = cv::Scalar(c[r],c[g],c[b]);
			}
		}
	}
}


cv::Scalar ColorGenerator::getColor()
{
    cv::Scalar cr = m_cr[m_index];
	m_index = (m_index + 1) % m_count;

	return cr;
}

cv::Scalar ColorGenerator::getColor(int idx)
{
    return m_cr[idx % m_count];
}

cv::Scalar ColorGenerator::makeColor(double x, double max)
{
    float colors[6][3] = { { 1, 0, 1 }, { 0, 0, 1 }, { 0, 1, 1 }, { 0, 1, 0 }, { 1, 1, 0 }, { 1, 0, 0 } };

    float ratio = (float)((x / max) * 5);
    int i = (int)floor(ratio);
    int j = (int)ceil(ratio);
    ratio -= i;
    float r = (1 - ratio) * colors[i][2] + ratio*colors[j][2];
    float g = (1 - ratio) * colors[i][1] + ratio*colors[j][1];
    float b = (1 - ratio) * colors[i][0] + ratio*colors[j][0];

    return cv::Scalar(b * 256, g * 256, r * 256);
}