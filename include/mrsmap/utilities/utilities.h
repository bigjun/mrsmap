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

#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <set>

#include <Eigen/Core>


namespace mrsmap {

	const int UNASSIGNED_LABEL = -1;
	const int OUTLIER_LABEL = -2;

	namespace colormapjet {
		double interpolate( double val, double y0, double x0, double y1, double x1 );
		double base( double val );
		double red( double gray );
		double green( double gray );
		double blue( double gray );
	}

	cv::Mat visualizeDepth( const cv::Mat& depthImg, float minDepth, float maxDepth );

	bool pointInImage( const Eigen::Vector4f& p );
	bool pointInImage( const Eigen::Vector4f& p, const unsigned int imageBorder );
	Eigen::Vector2f pointImagePos( const Eigen::Vector4f& p );

	void convertRGB2LAlphaBeta( float r, float g, float b, float& L, float& alpha, float& beta );
	void convertLAlphaBeta2RGB( float L, float alpha, float beta, float& r, float& g, float& b );
	void convertLAlphaBeta2RGBDamped( float L, float alpha, float beta, float& r, float& g, float& b );

	cv::Mat visualizeAlphaBetaPlane( float L, unsigned int imgSize );

	void imagesToPointCloud( const cv::Mat& depthImg, const cv::Mat& colorImg, const std::string& timeStamp, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, unsigned int downsampling = 1 );
	void imagesToPointCloudUnorganized( const cv::Mat& depthImg, const cv::Mat& colorImg, const std::string& timeStamp, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, unsigned int downsampling = 1 );

	void pointCloudToImage( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloud, cv::Mat& img );
	void pointCloudToImages( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloud, cv::Mat& img_rgb, cv::Mat& img_depth );

	void pointCloudsToOverlayImage( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& rgb_cloud, const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& overlay_cloud, cv::Mat& img );

	void downsamplePointCloud( const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloudIn, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloudOut, unsigned int downsampling );
	void downsamplePointCloud( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloudIn, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloudOut, unsigned int downsampling );

	double averageDepth( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloud );
	double medianDepth( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloud );

	void getCameraCalibration( cv::Mat& cameraMatrix, cv::Mat& distortionCoeffs );

	class SegmentationResult {
	public:
		SegmentationResult() {}
		~SegmentationResult() {}
//
		static void printConfusionMatrix( const std::map< unsigned int, std::map< int, unsigned int > >& confusionMatrix );

		// -1: unassigned label in result
		// -2: outlier label in result
//		std::map< unsigned int, std::map< int, unsigned int > > confusionMatrix;
		std::map< unsigned int, unsigned int > gtToResultColorMap;
		std::map< unsigned int, bool > objectInMotionGroup;

		double overallTPs, overallTPFPFNs;
		std::map< unsigned int, double > objectTPs, objectTPFPs, objectTPFNs, objectTPFPFNs;
	};

	void fuseSegmentationForMotionGroups( cv::Mat& img, const std::vector< std::vector< unsigned int > >& motionGroups );

	SegmentationResult compareToGroundTruth( cv::Mat& img_moseg, cv::Mat& img_gt, const std::vector< std::vector< unsigned int > >& motionGroups );

	unsigned int countPixels( cv::Mat& img, unsigned int color );

//	// combines labels according to motion groups
//	void applyMotionGroupsOnConfusionMatrix( SegmentationResult& result, const std::vector< std::vector< unsigned int > >& motionGroups );

//	double averagePixelAccuracy( const SegmentationResult& result, bool countOutliers );
//	double averageObjectAccuracy( const SegmentationResult& result, bool countOutliers );
//	double objectAccuracy( unsigned int objectId, const SegmentationResult& result, bool countOutliers );
//
//	std::pair< double, double > allObjectsAccuracyStats( const SegmentationResult& result, bool countOutliers );
//	std::pair< double, double > objectAccuracyStats( unsigned int objectId, const SegmentationResult& result, bool countOutliers );

	std::set< unsigned int > getObjectIds( cv::Mat& img_gt );

	void replaceObjectColor( cv::Mat& img_gt, unsigned int id_from, unsigned int id_to );

	unsigned int objectIdFromColori( unsigned int color );
	unsigned int objectIdFromColorf( float color );

	unsigned int colorForObjectId( unsigned int objectId );


	void colorForObjectClass( int c, float& r, float& g, float& b );



	void fillDepthFromRight( cv::Mat& imgDepth );
	void fillDepthFromLeft( cv::Mat& imgDepth );
	void fillDepthFromTop( cv::Mat& imgDepth );
	void fillDepthFromBottom( cv::Mat& imgDepth );


};



#endif /* UTILITIES_H_ */



