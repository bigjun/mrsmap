/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Computer Science Institute VI, University of Bonn
 *  Author: Joerg Stueckler, 24.09.2012
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of University of Bonn, Computer Science Institute 
 *     VI nor the names of its contributors may be used to endorse or 
 *     promote products derived from this software without specific 
 *     prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "mrsmap/utilities/utilities.h"
#include <boost/algorithm/string.hpp>

using namespace mrsmap;


double mrsmap::colormapjet::interpolate( double val, double y0, double x0, double y1, double x1 ) {
    return (val-x0)*(y1-y0)/(x1-x0) + y0;
}

double mrsmap::colormapjet::base( double val ) {
    if ( val <= -0.75 ) return 0;
    else if ( val <= -0.25 ) return interpolate( val, 0.0, -0.75, 1.0, -0.25 );
    else if ( val <= 0.25 ) return 1.0;
    else if ( val <= 0.75 ) return interpolate( val, 1.0, 0.25, 0.0, 0.75 );
    else return 0.0;
}

double mrsmap::colormapjet::red( double gray ) {
    return base( gray - 0.5 );
}

double mrsmap::colormapjet::green( double gray ) {
    return base( gray );
}

double mrsmap::colormapjet::blue( double gray ) {
    return base( gray + 0.5 );
}


cv::Mat mrsmap::visualizeDepth( const cv::Mat& depthImg, float minDepth, float maxDepth ) {

	cv::Mat img( depthImg.rows, depthImg.cols, CV_8UC3, 0.f );

	const float depthRange = maxDepth - minDepth;
	for( unsigned int y = 0; y < depthImg.rows; y++ ) {

		for( unsigned int x = 0; x < depthImg.cols; x++ ) {

			if( depthImg.at<unsigned short>(y,x) == 0 ) {
				img.at< cv::Vec3b >( y, x ) = cv::Vec3b( 255, 255, 255 );
			}
			else {

				float gray = 2.f * (depthImg.at<unsigned short>(y,x) - minDepth) / depthRange - 1.f;

				float rf = std::min( 1., std::max( 0., mrsmap::colormapjet::red( gray ) ) );
				float gf = std::min( 1., std::max( 0., mrsmap::colormapjet::green( gray ) ) );
				float bf = std::min( 1., std::max( 0., mrsmap::colormapjet::blue( gray ) ) );

				img.at< cv::Vec3b >( y, x ) = cv::Vec3b( 255 * bf, 255 * gf, 255 * rf );

			}

		}

	}

	return img;

}


Eigen::Vector2f mrsmap::pointImagePos( const Eigen::Vector4f& p ) {

	if( isnan( p(0) ) )
		return Eigen::Vector2f( p(0), p(0) );

	return Eigen::Vector2f( 525.0 * p(0) / p(2), 525.0 * p(1) / p(2) );

}


bool mrsmap::pointInImage( const Eigen::Vector4f& p ) {

	if( isnan( p(0) ) )
		return false;

	double px = 525.0 * p(0) / p(2);
	double py = 525.0 * p(1) / p(2);

	if( px < -320.0 || px > 320.0 || py < -240.0 || py > 240.0 ) {
		return false;
	}

	return true;

}

bool mrsmap::pointInImage( const Eigen::Vector4f& p, const unsigned int imageBorder ) {

	if( isnan( p(0) ) )
		return false;

	double px = 525.0 * p(0) / p(2);
	double py = 525.0 * p(1) / p(2);

	if( px < -320.0 + imageBorder || px > 320.0 - imageBorder || py < -240.0 + imageBorder || py > 240.0 - imageBorder ) {
		return false;
	}

	return true;

}

void mrsmap::convertRGB2LAlphaBeta( float r, float g, float b, float& L, float& alpha, float& beta ) {

	static const float sqrt305 = 0.5f*sqrtf(3);

	// RGB to L-alpha-beta:
	// normalize RGB to [0,1]
	// M := max( R, G, B )
	// m := min( R, G, B )
	// L := 0.5 ( M + m )
	// alpha := 0.5 ( 2R - G - B )
	// beta := 0.5 sqrt(3) ( G - B )
	L = 0.5f * ( std::max( std::max( r, g ), b ) + std::min( std::min( r, g ), b ) );
	alpha = 0.5f * ( 2.f*r - g - b );
	beta = sqrt305 * (g-b);

}

void mrsmap::convertLAlphaBeta2RGB( float L, float alpha, float beta, float& r, float& g, float& b ) {

	static const float pi3 = M_PI / 3.f;
	static const float pi3_inv = 1.f / pi3;

	// L-alpha-beta to RGB:
	// the mean should not lie beyond the RGB [0,1] range
	// sampled points could lie beyond, so we transform first to HSL,
	// "saturate" there, and then transform back to RGB
	// H = atan2(beta,alpha)
	// C = sqrt( alpha*alpha + beta*beta)
	// S = C / (1 - abs(2L-1))
	// saturate S' [0,1], L' [0,1]
	// C' = (1-abs(2L-1)) S'
	// X = C' (1- abs( (H/60) mod 2 - 1 ))
	// calculate luminance-free R' G' B'
	// m := L - 0.5 C
	// R, G, B := R1+m, G1+m, B1+m

	float h = atan2f( beta, alpha );
	float c = std::max( 0.f, std::min( 1.f, sqrtf( alpha*alpha + beta*beta ) ) );
	float s_norm = (1.f-fabsf(2.f*L - 1.f));
	float s = 0.f;
	if( s_norm > 1e-4f ) {
		s = std::max( 0.f, std::min( 1.f, c / s_norm ) );
		c = s_norm * s;
	}
	else
		c = 0.f;

	if( h < 0 )
		h += 2.f*M_PI;
	float h2 = pi3_inv * h;
	float h_sector = h2 - 2.f*floor(0.5f*h2);
	float x = c * (1.f-fabsf( h_sector-1.f ));

	float r1 = 0, g1 = 0, b1 = 0;
	if( h2 >= 0.f && h2 < 1.f )
		r1 = c, g1 = x;
	else if( h2 >= 1.f && h2 < 2.f )
		r1 = x, g1 = c;
	else if( h2 >= 2.f && h2 < 3.f )
		g1 = c, b1 = x;
	else if( h2 >= 3.f && h2 < 4.f )
		g1 = x, b1 = c;
	else if( h2 >= 4.f && h2 < 5.f )
		r1 = x, b1 = c;
	else
		r1 = c, b1 = x;

	float m = L - 0.5f * c;
	r = r1+m;
	b = b1+m;
	g = g1+m;

}


void mrsmap::convertLAlphaBeta2RGBDamped( float L, float alpha, float beta, float& r, float& g, float& b ) {

	static const float pi3 = M_PI / 3.f;
	static const float pi3_inv = 1.f / pi3;

	// L-alpha-beta to RGB:
	// the mean should not lie beyond the RGB [0,1] range
	// sampled points could lie beyond, so we transform first to HSL,
	// "saturate" there, and then transform back to RGB
	// H = atan2(beta,alpha)
	// C = sqrt( alpha*alpha + beta*beta)
	// S = C / (1 - abs(2L-1))
	// saturate S' [0,1], L' [0,1]
	// C' = (1-abs(2L-1)) S'
	// X = C' (1- abs( (H/60) mod 2 - 1 ))
	// calculate luminance-free R' G' B'
	// m := L - 0.5 C
	// R, G, B := R1+m, G1+m, B1+m

	float h = atan2f( beta, alpha );
	float c = std::max( 0.f, std::min( 1.f, sqrtf( alpha*alpha + beta*beta ) ) );
	float s_norm = (1.f-fabsf(2.f*L - 1.f));
	float s = 0.f;
	if( s_norm > 1e-4f ) {
		s = std::max( 0.f, std::min( 1.f, c / s_norm ) );
		// damp saturation stronger when lightness is bad
		s *= expf( -0.5f * 10.f * (L-0.5f) * (L-0.5f) );
		c = s_norm * s;
	}
	else
		c = 0.f;



	if( h < 0 )
		h += 2.f*M_PI;
	float h2 = pi3_inv * h;
	float h_sector = h2 - 2.f*floor(0.5f*h2);
	float x = c * (1.f-fabsf( h_sector-1.f ));

	float r1 = 0, g1 = 0, b1 = 0;
	if( h2 >= 0.f && h2 < 1.f )
		r1 = c, g1 = x;
	else if( h2 >= 1.f && h2 < 2.f )
		r1 = x, g1 = c;
	else if( h2 >= 2.f && h2 < 3.f )
		g1 = c, b1 = x;
	else if( h2 >= 3.f && h2 < 4.f )
		g1 = x, b1 = c;
	else if( h2 >= 4.f && h2 < 5.f )
		r1 = x, b1 = c;
	else
		r1 = c, b1 = x;

	float m = L - 0.5f * c;
	r = r1+m;
	b = b1+m;
	g = g1+m;

}


cv::Mat mrsmap::visualizeAlphaBetaPlane( float L, unsigned int imgSize ) {
	if( imgSize % 2 == 0 ) imgSize += 1;
	cv::Mat img( imgSize, imgSize, CV_8UC3, 0.f );

	const int radius = (imgSize-1) / 2;
	const float sqrt305 = 0.5f*sqrtf(3.f);

	for( int a = -radius; a <= radius; a++ ) {

		float alpha = (float)a / (float)radius;

		for( int b = -radius; b <= radius; b++ ) {

			float beta = (float)b / (float)radius;

			float rf, gf, bf;
			convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );
			img.at< cv::Vec3b >( radius-b, a+radius ) = cv::Vec3b( 255*bf, 255*gf, 255*rf );

		}

	}

	return img;
}



void mrsmap::imagesToPointCloud( const cv::Mat& depthImg, const cv::Mat& colorImg, const std::string& timeStamp, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, unsigned int downsampling ) {

	cloud->header.frame_id = "openni_rgb_optical_frame";
	cloud->is_dense = true;
	cloud->height = depthImg.rows / downsampling;
	cloud->width = depthImg.cols / downsampling;
	cloud->sensor_origin_ = Eigen::Vector4f( 0.f, 0.f, 0.f, 1.f );
	cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
	cloud->points.resize( cloud->height*cloud->width );

	const float invfocalLength = 1.f / 525.f;
	const float centerX = 319.5f;
	const float centerY = 239.5f;
	const float factor = 1.f / 5000.f;

	const unsigned short* depthdata = reinterpret_cast<const unsigned short*>( &depthImg.data[0] );
	const unsigned char* colordata = &colorImg.data[0];
	int idx = 0;
	for( unsigned int y = 0; y < depthImg.rows; y++ ) {
		for( unsigned int x = 0; x < depthImg.cols; x++ ) {

			if( x % downsampling != 0 || y % downsampling != 0 ) {
				colordata += 3;
				depthdata++;
				continue;
			}

			pcl::PointXYZRGB& p = cloud->points[idx];

			if( *depthdata == 0 ) { //|| factor * (float)(*depthdata) > 10.f ) {
				p.x = std::numeric_limits<float>::quiet_NaN();
				p.y = std::numeric_limits<float>::quiet_NaN();
				p.z = std::numeric_limits<float>::quiet_NaN();
			}
			else {
				float xf = x;
				float yf = y;
				float dist = factor * (float)(*depthdata);
				p.x = (xf-centerX) * dist * invfocalLength;
				p.y = (yf-centerY) * dist * invfocalLength;
				p.z = dist;
			}

			depthdata++;

			int b = (*colordata++);
			int g = (*colordata++);
			int r = (*colordata++);

			int rgb = ( r << 16 ) + ( g << 8 ) + b;
			p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

			idx++;


		}
	}

}


double mrsmap::averageDepth( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloud ) {

	double sum = 0.0;
	double num = 0.0;
	int idx = 0;
	for( unsigned int y = 0; y < cloud->height; y++ ) {
		for( unsigned int x = 0; x < cloud->width; x++ ) {

			const pcl::PointXYZRGB& p = cloud->points[idx];

			if( !isnan( p.z ) ) {
				sum += p.z;
				num += 1.0;
			}

			idx++;

		}
	}

	return sum / num;

}


double mrsmap::medianDepth( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloud ) {

	std::vector< double > depths;
	int idx = 0;
	for( unsigned int y = 0; y < cloud->height; y++ ) {
		for( unsigned int x = 0; x < cloud->width; x++ ) {

			const pcl::PointXYZRGB& p = cloud->points[idx];

			if( !isnan( p.z ) ) {
				depths.push_back(p.z);
			}

			idx++;

		}
	}

	std::sort( depths.begin(), depths.end() );
	depths.push_back(0);
	return depths[depths.size()/2];

}


void mrsmap::imagesToPointCloudUnorganized( const cv::Mat& depthImg, const cv::Mat& colorImg, const std::string& timeStamp, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, unsigned int downsampling ) {

	cloud->header.frame_id = "openni_rgb_optical_frame";
	cloud->is_dense = false;
	cloud->sensor_origin_ = Eigen::Vector4f( 0.f, 0.f, 0.f, 1.f );
	cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
	cloud->points.reserve( cloud->height*cloud->width );

	const float invfocalLength = 1.f / 525.f;
	const float centerX = 319.5f;
	const float centerY = 239.5f;
	const float factor = 1.f / 5000.f;

	const unsigned short* depthdata = reinterpret_cast<const unsigned short*>( &depthImg.data[0] );
	const unsigned char* colordata = &colorImg.data[0];
	int idx = 0;
	for( unsigned int y = 0; y < depthImg.rows; y++ ) {
		for( unsigned int x = 0; x < depthImg.cols; x++ ) {

			if( x % downsampling != 0 || y % downsampling != 0 ) {
				colordata += 3;
				depthdata++;
				continue;
			}

			pcl::PointXYZRGB p;

			if( *depthdata == 0 || factor * (float)(*depthdata) > 5.f ) {
			}
			else {
				float xf = x;
				float yf = y;
				float dist = factor * (float)(*depthdata);
				p.x = (xf-centerX) * dist * invfocalLength;
				p.y = (yf-centerY) * dist * invfocalLength;
				p.z = dist;
			}

			depthdata++;

			int b = (*colordata++);
			int g = (*colordata++);
			int r = (*colordata++);

			int rgb = ( r << 16 ) + ( g << 8 ) + b;
			p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

			cloud->push_back( p );

			idx++;


		}
	}

	cloud->width = cloud->points.size();

}


void mrsmap::getCameraCalibration( cv::Mat& cameraMatrix, cv::Mat& distortionCoeffs ) {

	distortionCoeffs = cv::Mat( 1, 5, CV_32FC1, 0.f );
	cameraMatrix = cv::Mat( 3, 3, CV_32FC1, 0.f );

	cameraMatrix.at<float>(0,0) = 525.f;
	cameraMatrix.at<float>(1,1) = 525.f;
	cameraMatrix.at<float>(2,2) = 1.f;

	cameraMatrix.at<float>(0,2) = 319.5f;
	cameraMatrix.at<float>(1,2) = 239.5f;

}


void mrsmap::pointCloudToImage( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloud, cv::Mat& img ) {

	img = cv::Mat( cloud->height, cloud->width, CV_8UC3, 0.f );

	int idx = 0;
	for( unsigned int y = 0; y < cloud->height; y++ ) {
		for( unsigned int x = 0; x < cloud->width; x++ ) {

			const pcl::PointXYZRGB& p = cloud->points[idx];

			cv::Vec3b px;
			px[0] = p.b;
			px[1] = p.g;
			px[2] = p.r;

			img.at< cv::Vec3b >( y, x ) = px;

			idx++;

		}
	}

}


void mrsmap::pointCloudToImages( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloud, cv::Mat& img_rgb, cv::Mat& img_depth ) {

	img_rgb = cv::Mat( cloud->height, cloud->width, CV_8UC3, 0.f );
	img_depth = cv::Mat( cloud->height, cloud->width, CV_32FC1, 0.f );

	int idx = 0;
	for( unsigned int y = 0; y < cloud->height; y++ ) {
		for( unsigned int x = 0; x < cloud->width; x++ ) {

			const pcl::PointXYZRGB& p = cloud->points[idx];

			cv::Vec3b px;
			px[0] = p.b;
			px[1] = p.g;
			px[2] = p.r;

			img_rgb.at< cv::Vec3b >( y, x ) = px;
			img_depth.at< float >( y, x ) = p.z;

			idx++;

		}
	}

}


void mrsmap::pointCloudsToOverlayImage( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& rgb_cloud, const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& overlay_cloud, cv::Mat& img ) {

	img = cv::Mat( rgb_cloud->height, rgb_cloud->width, CV_8UC3, 0.f );

	float alpha = 0.2;

	int idx = 0;
	for( unsigned int y = 0; y < rgb_cloud->height; y++ ) {
		for( unsigned int x = 0; x < rgb_cloud->width; x++ ) {

			const pcl::PointXYZRGB& p1 = rgb_cloud->points[idx];
			const pcl::PointXYZRGB& p2 = overlay_cloud->points[idx];

			cv::Vec3b px;
			px[0] = (1-alpha) * p1.b + alpha * p2.b;
			px[1] = (1-alpha) * p1.g + alpha * p2.g;
			px[2] = (1-alpha) * p1.r + alpha * p2.r;

			img.at< cv::Vec3b >( y, x ) = px;

			idx++;

		}
	}

}



void mrsmap::downsamplePointCloud( const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloudIn, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloudOut, unsigned int downsampling ) {

	cloudOut = pcl::PointCloud< pcl::PointXYZRGB >::Ptr( new pcl::PointCloud< pcl::PointXYZRGB >() );

	cloudOut->header = cloudIn->header;
	cloudOut->is_dense = cloudIn->is_dense;
	cloudOut->width = cloudIn->width / downsampling;
	cloudOut->height = cloudIn->height / downsampling;
	cloudOut->sensor_origin_ = cloudIn->sensor_origin_;
	cloudOut->sensor_orientation_ = cloudIn->sensor_orientation_;

	cloudOut->points.resize( cloudOut->width*cloudOut->height );

	unsigned int idx = 0;
	for( unsigned int y = 0; y < cloudIn->height; y++ ) {

		if( y % downsampling != 0 )
			continue;

		for( unsigned int x = 0; x < cloudIn->width; x++ ) {

			if( x % downsampling != 0 )
				continue;

			cloudOut->points[idx++] = cloudIn->points[ y*cloudIn->width + x ];

		}
	}



}


void mrsmap::downsamplePointCloud( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloudIn, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloudOut, unsigned int downsampling ) {

	cloudOut = pcl::PointCloud< pcl::PointXYZRGB >::Ptr( new pcl::PointCloud< pcl::PointXYZRGB >() );

	cloudOut->header = cloudIn->header;
	cloudOut->is_dense = cloudIn->is_dense;
	cloudOut->width = cloudIn->width / downsampling;
	cloudOut->height = cloudIn->height / downsampling;
	cloudOut->sensor_origin_ = cloudIn->sensor_origin_;
	cloudOut->sensor_orientation_ = cloudIn->sensor_orientation_;

	cloudOut->points.resize( cloudOut->width*cloudOut->height );

	unsigned int idx = 0;
	for( unsigned int y = 0; y < cloudIn->height; y++ ) {

		if( y % downsampling != 0 )
			continue;

		for( unsigned int x = 0; x < cloudIn->width; x++ ) {

			if( x % downsampling != 0 )
				continue;

			cloudOut->points[idx++] = cloudIn->points[ y*cloudIn->width + x ];

		}
	}



}



std::set< unsigned int > mrsmap::getObjectIds( cv::Mat& img_gt ) {

	std::set< unsigned int > ids;

	for( unsigned int y = 0; y < img_gt.rows; y++ ) {

		for( unsigned int x = 0; x < img_gt.cols; x++ ) {

			cv::Vec3b& p_gt = img_gt.at< cv::Vec3b >(y,x);

			unsigned int h_gt = (p_gt[2] << 16) | (p_gt[1] << 8) | p_gt[0];

			// dont care labels
			if( h_gt == 0xFFFFFF ) {
				continue;
			}

			ids.insert( objectIdFromColori( h_gt ) );

		}

	}

	return ids;

}


void mrsmap::replaceObjectColor( cv::Mat& img_gt, unsigned int id_from, unsigned int id_to ) {

	unsigned int c_from = colorForObjectId( id_from );
	unsigned int c_to = colorForObjectId( id_to );

	for( unsigned int y = 0; y < img_gt.rows; y++ ) {
		for( unsigned int x = 0; x < img_gt.cols; x++ ) {

			cv::Vec3b& p_gt = img_gt.at< cv::Vec3b >(y,x);

			unsigned int h_gt = (p_gt[2] << 16) | (p_gt[1] << 8) | p_gt[0];

			// dont care labels
			if( h_gt == 0xFFFFFF ) {
				continue;
			}

			if( h_gt == c_from ) {
				p_gt[0] = c_to & 0xFF;
				p_gt[1] = (c_to >> 8) & 0xFF;
				p_gt[2] = (c_to >> 16) & 0xFF;
			}

		}
	}

}


void mrsmap::fuseSegmentationForMotionGroups( cv::Mat& img, const std::vector< std::vector< unsigned int > >& motionGroupsIn ) {

	std::vector< std::vector< unsigned int > > motionGroups = motionGroupsIn;

	// replace object id colors with color for lowest object id in same motion group
	for( unsigned int i = 0; i < motionGroups.size(); i++ ) {

		// use lowest index for the group
		std::sort( motionGroups[i].begin(), motionGroups[i].end() );

		for( unsigned int j = 0; j < motionGroups[i].size(); j++ ) {

			replaceObjectColor( img, motionGroups[i][j], motionGroups[i][0] );

		}
	}

}


SegmentationResult mrsmap::compareToGroundTruth( cv::Mat& img_result, cv::Mat& img_gt, const std::vector< std::vector< unsigned int > >& motionGroups ) {

	SegmentationResult result;

	// consider motion groups
	cv::Mat img_gt_mod = img_gt.clone();
	fuseSegmentationForMotionGroups( img_gt_mod, motionGroups );

	// compare each pixel in gt image with segmentation to extract confusion matrix

	// let h_gt correspond to h_result
	// tp: gt( s ) == h_gt && res( s ) == h_result
	// fp: gt( s ) != h_gt && res( s ) == h_result
	// tn: gt( s ) != h_gt && res( s ) != h_result
	// fn: gt( s ) == h_gt && res( s ) != h_result


//	// we do not know correspondences between segments yet
//	std::map< unsigned int, std::map< int, unsigned int > > unassignedConfusionMatrix;
//	std::map< unsigned int, bool > gtColorsAssigned, resColorsAssigned;
//
//	for( unsigned int y = 0; y < img_gt_mod.rows; y++ ) {
//
//		for( unsigned int x = 0; x < img_gt_mod.cols; x++ ) {
//
//			cv::Vec3b& p_gt = img_gt_mod.at< cv::Vec3b >(y,x);
//			cv::Vec3b& p_seg = img_result.at< cv::Vec3b >(y,x);
//
//			unsigned int h_gt = (p_gt[2] << 16) | (p_gt[1] << 8) | p_gt[0];
//			unsigned int h_seg = (p_seg[2] << 16) | (p_seg[1] << 8) | p_seg[0];
//
//			// dont care labels
//			if( h_gt == 0xFFFFFF ) {
//				continue;
//			}
//
//			if( h_seg == 0xFFFFFF ) {
//				continue;
//			}
//
//			unassignedConfusionMatrix[h_gt][h_seg] = 0;
//
//			gtColorsAssigned[h_gt] = false;
//
//			if( h_seg != 0xFF0000 )
//				resColorsAssigned[h_seg] = false;
//
//		}
//
//	}
//
//
//	for( unsigned int y = 0; y < img_gt_mod.rows; y++ ) {
//
//		for( unsigned int x = 0; x < img_gt_mod.cols; x++ ) {
//
//			cv::Vec3b& p_gt = img_gt_mod.at< cv::Vec3b >(y,x);
//			cv::Vec3b& p_seg = img_result.at< cv::Vec3b >(y,x);
//
//			unsigned int h_gt = (p_gt[2] << 16) | (p_gt[1] << 8) | p_gt[0];
//			unsigned int h_seg = (p_seg[2] << 16) | (p_seg[1] << 8) | p_seg[0];
//
//			// dont care labels
//			if( h_gt == 0xFFFFFF ) {
//				continue;
//			}
//
//			if( h_seg == 0xFFFFFF ) {
//				continue;
//			}
//
//			unassignedConfusionMatrix[h_gt][h_seg] += 1; // tp, if h_gt would correspond to h_seg
//
//		}
//
//	}
//
//
//	for( std::map< unsigned int, std::map< int, unsigned int > >::iterator it = unassignedConfusionMatrix.begin(); it != unassignedConfusionMatrix.end(); ++it ) {
//
//		unsigned int objectId = objectIdFromColori( it->first );
//		result.objectInMotionGroup[objectId] = false;
//		for( unsigned int i = 0; i < motionGroups.size(); i++ ) {
//			if( motionGroups[i].size() > 1 && std::find( motionGroups[i].begin(), motionGroups[i].end(), objectId ) != motionGroups[i].end() ) {
//				result.objectInMotionGroup[objectId] = true;
//			}
//		}
//
//		std::cout << "obj " << objectId << (result.objectInMotionGroup[objectId] ? " is in motion group\n" : " is not in motion group\n");
//
//	}
//
//
//	// greedy bijective mapping between result colors and gt colors
//	std::map< unsigned int, unsigned int > resultToGTColorMap;
//	unsigned int assignedGTs = 0;
//	unsigned int assignedRes = 0;
//	while( assignedGTs < gtColorsAssigned.size() && assignedRes < resColorsAssigned.size() ) {
//
//		unsigned int bestSegGTColor = 0;
//		unsigned int bestSegResColor = 0;
//		double bestTPs = -std::numeric_limits<double>::max();
//
//		for( std::map< unsigned int, std::map< int, unsigned int > >::iterator it = unassignedConfusionMatrix.begin(); it != unassignedConfusionMatrix.end(); ++it ) {
//
//			if( gtColorsAssigned[ it->first ] )
//				continue;
//
//			// assign largest overlapping segment in result to gt label
//
//			for( std::map< int, unsigned int >::iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2 ) {
//
//				if( resColorsAssigned[ it2->first ] )
//					continue;
//
//				// dont consider outlier label
//				if( it2->first == 0xFF0000 )
//					continue;
//
//				if( it2->second > bestTPs ) {
//					bestSegGTColor = it->first;
//					bestSegResColor = it2->first;
//					bestTPs = it2->second;
//				}
//
//			}
//
//		}
//
//		if( bestTPs >= 0.0 ) {
//			resultToGTColorMap[bestSegResColor] = bestSegGTColor;
//			result.gtToResultColorMap[bestSegGTColor] = bestSegResColor;
//		}
//
//		resColorsAssigned[bestSegResColor] = true;
//		gtColorsAssigned[bestSegGTColor] = true;
//
//		assignedGTs++;
//		assignedRes++;
//
//		unsigned int r = (bestSegGTColor >> 16) & 0xFF;
//		unsigned int g = (bestSegGTColor >> 8) & 0xFF;
//		unsigned int b = bestSegGTColor & 0xFF;
//
//		unsigned int rs = (bestSegResColor >> 16) & 0xFF;
//		unsigned int gs = (bestSegResColor >> 8) & 0xFF;
//		unsigned int bs = bestSegResColor & 0xFF;
//
//		std::cout << "gt label " << r << " " << g << " " << b << " has best overlap " << bestTPs << " for res color: " << rs << " " << gs << " " << bs << "\n";
//
//	}
//
//
////	SegmentationResult::printConfusionMatrix( unassignedConfusionMatrix );
//
//	// fill confusion matrix using assignments and counts
//	for( std::map< unsigned int, std::map< int, unsigned int > >::iterator it = unassignedConfusionMatrix.begin(); it != unassignedConfusionMatrix.end(); ++it ) {
//
//		int gtObjectId = objectIdFromColori( it->first );
//
//		// initialize each entry in confusion matrix for current gt label with 0
//		for( std::map< unsigned int, std::map< int, unsigned int > >::iterator it2 = unassignedConfusionMatrix.begin(); it2 != unassignedConfusionMatrix.end(); ++it2 ) {
//			int gtObjectId2 = objectIdFromColori( it2->first );
//			result.confusionMatrix[gtObjectId][gtObjectId2] = 0;
//		}
//		result.confusionMatrix[gtObjectId][UNASSIGNED_LABEL] = 0;
//		result.confusionMatrix[gtObjectId][OUTLIER_LABEL] = 0;
//
//		for( std::map< int, unsigned int >::iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2 ) {
//
//			// result label assigned?
//			int resObjectId = UNASSIGNED_LABEL; // -1 is id for non-assigned labels
//			if( it2->first == 0xFF0000 ) {
//				resObjectId = OUTLIER_LABEL; // outlier label
//			}
//			else if( resultToGTColorMap.find( it2->first ) != resultToGTColorMap.end() ) {
//				resObjectId = objectIdFromColori( resultToGTColorMap[ it2->first ] );
//			}
//
//			result.confusionMatrix[gtObjectId][resObjectId] += it2->second;
//
//		}
//
//	}
//
//	return result;


	std::map< unsigned int, std::map< unsigned int, unsigned int > > truepositives_map;
	std::map< unsigned int, unsigned int > counted_gt_sites, counted_seg_sites;
	for( unsigned int y = 0; y < img_gt_mod.rows; y++ ) {

		for( unsigned int x = 0; x < img_gt_mod.cols; x++ ) {

			cv::Vec3b& p_gt = img_gt_mod.at< cv::Vec3b >(y,x);
			cv::Vec3b& p_seg = img_result.at< cv::Vec3b >(y,x);

			unsigned int h_gt = (p_gt[2] << 16) | (p_gt[1] << 8) | p_gt[0];
			unsigned int h_seg = (p_seg[2] << 16) | (p_seg[1] << 8) | p_seg[0];

			// dont care labels
			if( h_gt == 0xFFFFFF ) {
				continue;
			}

			if( h_seg == 0xFFFFFF ) {
				continue;
			}

			truepositives_map[h_gt][h_seg] += 1; // tp if h_gt would correspond to h_seg
			counted_gt_sites[h_gt] += 1; // this is (tp+fn) for any h_seg that would correspond to h_gt
			counted_seg_sites[h_seg] += 1; // this is (tp+fp) for any h_gt that would correspond to h_seg

		}

	}

	for( std::map< unsigned int, std::map< unsigned int, unsigned int > >::iterator it = truepositives_map.begin(); it != truepositives_map.end(); ++it ) {

		unsigned int objectId = objectIdFromColori( it->first );
		result.objectInMotionGroup[objectId] = false;
		for( unsigned int i = 0; i < motionGroups.size(); i++ ) {
			if( motionGroups[i].size() > 1 && std::find( motionGroups[i].begin(), motionGroups[i].end(), objectId ) != motionGroups[i].end() ) {
				result.objectInMotionGroup[objectId] = true;
			}
		}

		std::cout << "obj " << objectId << (result.objectInMotionGroup[objectId] ? " is in motion group\n" : " is not in motion group\n");

		if( objectId == 0 || !result.objectInMotionGroup[objectId] ) {

			result.objectTPFPs[objectId] = 0;
			result.objectTPFNs[objectId] = 0;
			result.objectTPs[objectId] = 0;
			result.objectTPFPFNs[objectId] = 0;

		}

	}

	// greedy assignment of gt-result segment pairs
//	std::map< unsigned int, std::map< unsigned int, double > > segAccuracy;

//	std::map< unsigned int, std::pair< unsigned int, double > > bestSegAccuracyMap;
//	std::map< unsigned int, std::pair< unsigned int, double > > bestTPsMap;
//	std::map< unsigned int, std::pair< unsigned int, double > > bestTPFPFNsMap;

	result.overallTPs = 0.0; // counts every pixel whether or not it is in a motion group or not
	result.overallTPFPFNs = 0.0; // counts every pixel whether or not it is in a motion group or not


	for( std::map< unsigned int, std::map< unsigned int, unsigned int > >::iterator it = truepositives_map.begin(); it != truepositives_map.end(); ++it ) {

		unsigned int bestSegResColor = 0;
		double bestSegAccuracy = 0.0;
		double bestTPs = 0.0;
		double bestTPFPs = 0.0;
		double bestTPFPFNs = 0.0;
		for( std::map< unsigned int, unsigned int >::iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2 ) {

			// dont consider outlier label
			if( it2->first == 0xFF0000 )
				continue;

			// tp / (tp+fp+fn) = tp / (tp+fn + tp+fp - tp)
			double seg_accuracy = ((double)it2->second) / (double)( counted_gt_sites[it->first] + counted_seg_sites[it2->first] - it2->second );
//			segAccuracy[it->first][it2->first] = seg_accuracy;

			if( seg_accuracy > bestSegAccuracy ) {
				bestSegAccuracy = seg_accuracy;
				bestSegResColor = it2->first;

				bestTPs = it2->second;
				bestTPFPs = counted_seg_sites[it2->first];
				bestTPFPFNs = counted_gt_sites[it->first] + counted_seg_sites[it2->first] - it2->second;
			}

		}

		unsigned int r = (it->first >> 16) & 0xFF;
		unsigned int g = (it->first >> 8) & 0xFF;
		unsigned int b = it->first & 0xFF;

//		bestSegAccuracyMap[ it->first ] = std::pair< unsigned int, double >( bestSegResColor, bestSegAccuracy );
//		bestTPsMap[ it->first ] = std::pair< unsigned int, double >( bestSegResColor, bestTPs );
//		bestTPFPFNsMap[ it->first ] = std::pair< unsigned int, double >( bestSegResColor, bestTPFPFNs );

		result.overallTPs += bestTPs;
		result.overallTPFPFNs += bestTPFPFNs;

		unsigned int objectId = objectIdFromColori( it->first );
		if( objectId == 0 || !result.objectInMotionGroup[objectId] ) {
			result.objectTPFNs[objectId] = counted_gt_sites[it->first];
			result.objectTPFPs[objectId] = bestTPFPs;
			result.objectTPs[objectId] = bestTPs;
			result.objectTPFPFNs[objectId] = bestTPFPFNs;
		}

		unsigned int rs = (bestSegResColor >> 16) & 0xFF;
		unsigned int gs = (bestSegResColor >> 8) & 0xFF;
		unsigned int bs = bestSegResColor & 0xFF;

		result.gtToResultColorMap[it->first] = bestSegResColor;

		std::cout << "gt label " << r << " " << g << " " << b << " has best segmentation accuracy " << bestSegAccuracy << " for res color: " << rs << " " << gs << " " << bs << "\n";

	}


	return result;


}


unsigned int mrsmap::countPixels( cv::Mat& img, unsigned int color ) {

	unsigned int count = 0;

	for( unsigned int y = 0; y < img.rows; y++ ) {

		for( unsigned int x = 0; x < img.cols; x++ ) {

			cv::Vec3b& p = img.at< cv::Vec3b >(y,x);

			unsigned int h = (p[2] << 16) | (p[1] << 8) | p[0];

			// dont care labels
			if( h == color ) {
				count++;
			}

		}

	}

	return count;

}


//void mrsmap::SegmentationResult::printConfusionMatrix( const std::map< unsigned int, std::map< int, unsigned int > >& confusionMatrix ) {
//
//	for( std::map< unsigned int, std::map< int, unsigned int > >::const_iterator it = confusionMatrix.begin(); it != confusionMatrix.end(); ++it ) {
//
//		std::cout << it->first << ": ";
//		for( std::map< int, unsigned int >::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2 ) {
//
//			std::cout << it2->second << "(" << it2->first << ") ";
//
//		}
//
//		std::cout << "\n";
//
//	}
//
//}


//void mrsmap::applyMotionGroupsOnConfusionMatrix( SegmentationResult& result, const std::vector< std::vector< unsigned int > >& motionGroups ) {
//
//	for( std::map< unsigned int, std::map< int, unsigned int > >::iterator it = result.confusionMatrix.begin(); it != result.confusionMatrix.end(); ++it ) {
//
//		// only preserve one label from each motion group
//		for( unsigned int i = 0; i < motionGroups.size(); i++ ) {
//
//			if( it->second.find( motionGroups[i][0] ) != it->second.end() ) {
//				for( unsigned int j = 1; j < motionGroups[i].size(); j++ ) {
//					it->second.erase( motionGroups[i][j] );
//				}
//
//				if( it->first != motionGroups[i][0] && std::find( motionGroups[i].begin(), motionGroups[i].end(), it->first ) != motionGroups[i].end() ) {
//					it->second[it->first] = it->second[motionGroups[i][0]];
//					it->second.erase( motionGroups[i][0] );
//				}
//			}
//
//		}
//
//	}
//
//}


//double mrsmap::averagePixelAccuracy( const SegmentationResult& result, bool countOutliers ) {
//
//	double TPs = 0.0;
//	double TPFPFNs = 0.0;
//	for( std::map< unsigned int, std::map< int, unsigned int > >::const_iterator it = result.confusionMatrix.begin(); it != result.confusionMatrix.end(); ++it ) {
//		for( std::map< int, unsigned int >::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2 ) {
//
//			if( !countOutliers && it2->first == OUTLIER_LABEL )
//				continue;
//
//			if( it->first == it2->first ) {
//				TPs += it2->second;
//			}
//
//			TPFPFNs += it2->second;
//
//		}
//	}
//
//	return TPs / TPFPFNs;
//
//}
//
//std::pair< double, double > mrsmap::allObjectsAccuracyStats( const SegmentationResult& result, bool countOutliers ) {
//
//	// tp / (tp+fp+fn)
//
//	double TPs = 0.0;
//	double TPFPFNs = 0.0;
//
//	for( std::map< unsigned int, std::map< int, unsigned int > >::const_iterator it = result.confusionMatrix.begin(); it != result.confusionMatrix.end(); ++it ) {
//
//		// allow background segment to be in motion grouping with others
//		if( it->first != 0 && result.objectInMotionGroup.find(it->first)->second )
//			continue;
//
//
//		for( std::map< int, unsigned int >::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2 ) {
//
//			if( !countOutliers && it2->first == OUTLIER_LABEL )
//				continue;
//
//			if( it->first == it2->first ) {
//				TPs += it2->second;
//			}
//
//			TPFPFNs += it2->second; // tp+fp so far
//
//		}
//
//
//		for( std::map< unsigned int, std::map< int, unsigned int > >::const_iterator it2 = result.confusionMatrix.begin(); it2 != result.confusionMatrix.end(); ++it2 ) {
//
//			if( it->first == it2->first )
//				continue;
//
//			TPFPFNs += it2->second.find(it->first)->second;
//
//		}
//
//
////		acc += TPs / TPFPFNs;
////		numObjects += 1.0;
//
//	}
//
//	return std::pair< double, double >( TPs, TPFPFNs );
//
//}
//
//std::pair< double, double > mrsmap::objectAccuracyStats( unsigned int objectId, const SegmentationResult& result, bool countOutliers ) {
//
//	std::map< unsigned int, std::map< int, unsigned int > >::const_iterator it = result.confusionMatrix.find( objectId );
//	if( it == result.confusionMatrix.end() )
//		return 0.0;
//
//	double TPs = 0.0;
//	double TPFPFNs = 0.0;
//
//	for( std::map< int, unsigned int >::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2 ) {
//
//		if( !countOutliers && it2->first == OUTLIER_LABEL )
//			continue;
//
//		if( objectId == it2->first ) {
//			TPs += it2->second;
//		}
//
//		TPFPFNs += it2->second;
//
//	}
//
//	for( std::map< unsigned int, std::map< int, unsigned int > >::const_iterator it2 = result.confusionMatrix.begin(); it2 != result.confusionMatrix.end(); ++it2 ) {
//
//		if( objectId == it2->first )
//			continue;
//
//		TPFPFNs += it2->second.find(it->first)->second;
//
//	}
//
//	return std::pair< double, double >( TPs, TPFPFNs );
//
//}
//
//
//double mrsmap::averageObjectAccuracy( const SegmentationResult& result, bool countOutliers ) {
//
//	std::pair< double, double > stats = allObjectsAccuracyStats( result, countOutliers );
//	return stats.first / stats.second;
//
//}
//
//
//double mrsmap::objectAccuracy( unsigned int objectId, const SegmentationResult& result, bool countOutliers ) {
//
//	std::pair< double, double > stats = objectAccuracyStats( objectId, result, countOutliers );
//	return stats.first / stats.second;
//
//}


unsigned int mrsmap::objectIdFromColori( unsigned int color ) {

	switch( color ) {

	case 0x0000FF: // blue
		return 0;

	case 0xFFFF00: // yellow
		return 1;

	case 0xFF0000: // red
		return 2;

	case 0x00FFFF: // cyan
		return 3;

	case 0xFF00FF: // magenta
		return 4;

	}

	if( color != 0xFFFFFF ) {
		std::cout << "SOMETHING IS WRONG WITH THE COLORS: " << std::hex << color << "\n";
		exit(-1);
	}

	return 0;

}


unsigned int mrsmap::objectIdFromColorf( float color ) {

	unsigned int colori = *reinterpret_cast< unsigned int* >( &color );

	return objectIdFromColori( colori );

}


unsigned int mrsmap::colorForObjectId( unsigned int objectId ) {

	switch( objectId ) {

	case 0: // blue
		return 0x0000FF;

	case 1: // yellow
		return 0xFFFF00;

	case 2: // red
		return 0xFF0000;

	case 3: // vyan
		return 0x00FFFF;

	case 4: // magenta
		return 0xFF00FF;

	}

	return 0;

}



void mrsmap::colorForObjectClass( int c, float& r, float& g, float& b ) {

	switch( c ) {
	case 0:
		r = 1;
		g = 0;
		b = 0;
		break;
	case 1:
		r = 0;
		g = 1;
		b = 0;
		break;
	case 2:
		r = 0;
		g = 0;
		b = 1;
		break;
	case 3:
		r = 1;
		g = 1;
		b = 0;
		break;
	case 4:
		r = 1;
		g = 0;
		b = 1;
		break;
	case 5:
		r = 0;
		g = 1;
		b = 1;
		break;
	default:
		r = 1;
		g = 1;
		b = 1;
		break;
	};
}


void mrsmap::fillDepthFromRight( cv::Mat& imgDepth ) {

	for( unsigned int y = 0; y < imgDepth.rows; y++ ) {

		for( int x = imgDepth.cols-2; x >= 0; x-- ) {

			unsigned short& d = imgDepth.at< unsigned short >( y, x );
			if( d == 0 )
				d = imgDepth.at< unsigned short >( y, x+1 );

		}

	}

}


void mrsmap::fillDepthFromLeft( cv::Mat& imgDepth ) {

	for( unsigned int y = 0; y < imgDepth.rows; y++ ) {

		for( unsigned int x = 1; x < imgDepth.cols; x++ ) {

			unsigned short& d = imgDepth.at< unsigned short >( y, x );
			if( d == 0 )
				d = imgDepth.at< unsigned short >( y, x-1 );

		}

	}

}


void mrsmap::fillDepthFromTop( cv::Mat& imgDepth ) {

	for( unsigned int y = 1; y < imgDepth.rows; y++ ) {

		for( unsigned int x = 0; x < imgDepth.cols; x++ ) {

			unsigned short& d = imgDepth.at< unsigned short >( y, x );
			if( d == 0 )
				d = imgDepth.at< unsigned short >( y-1, x );

		}

	}

}


void mrsmap::fillDepthFromBottom( cv::Mat& imgDepth ) {

	for( int y = imgDepth.rows-2; y >= 0; y-- ) {

		for( unsigned int x = 0; x < imgDepth.cols; x++ ) {

			unsigned short& d = imgDepth.at< unsigned short >( y, x );
			if( d == 0 )
				d = imgDepth.at< unsigned short >( y+1, x );

		}

	}

}





