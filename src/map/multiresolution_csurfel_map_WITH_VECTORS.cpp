/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Computer Science Institute VI, University of Bonn
 *  Author: Joerg Stueckler, 02.05.2011
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

#include "mrsmap/map/multiresolution_csurfel_map.h"

#include <mrsmap/utilities/utilities.h>

#include "octreelib/feature/normalestimation.h"
#include "octreelib/algorithm/downsample.h"

#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"

#include <ostream>
#include <fstream>


using namespace mrsmap;

#define MAX_VIEWDIR_DIST cos( 0.25 * M_PI + 0.125*M_PI )


template <typename T>
gsl_rng* MultiResolutionColorSurfelMap<T>::r = NULL;

template <typename T>
MultiResolutionColorSurfelMap<T>::Params::Params() {

	dist_dependency = 0.01f;

	depthNoiseFactor = 1.f;
	pixelNoise = 5.f;
//	depthNoiseAssocFactor = 4.f;
//	pixelNoiseAssocFactor = 4.f;
	usePointFeatures = false;
	debugPointFeatures = false;
	GridCols = 8;
	GridRows = 6;
	numPointFeatures = 4000;
	GridCellMax = 25;

}


template <typename T>
MultiResolutionColorSurfelMap<T>::ImagePreAllocator::ImagePreAllocator()
: imageNodeAllocator_( 20000 ) {
	imgKeys = NULL;
	valueMap = NULL;
	node_image_ = NULL;
	width = height = 0;
}


template <typename T>
MultiResolutionColorSurfelMap<T>::ImagePreAllocator::~ImagePreAllocator() {
	if( imgKeys )
		delete[] imgKeys;

	if( valueMap )
		delete[] valueMap;
}

template <typename T>
void MultiResolutionColorSurfelMap<T>::ImagePreAllocator::prepare( unsigned int w, unsigned int h, bool buildNodeImage ) {

	typedef T* NodeValuePtr;
	typedef spatialaggregate::OcTreeNode< float, T >* NodePtr;

	if( !valueMap || height != h || width != w ) {

		if( imgKeys )
			delete[] imgKeys;
		imgKeys = new uint64_t[ w*h ];

		if( valueMap )
			delete[] valueMap;

		valueMap = new NodeValuePtr[ w*h ];

		if( node_image_ )
			delete[] node_image_;

		if( buildNodeImage )
			node_image_ = new NodePtr[w*h];

		infoList.resize( w*h );

		width = w;
		height = h;

	}

	memset( &imgKeys[0], 0LL, w*h * sizeof( uint64_t ) );
	memset( &valueMap[0], 0, w*h * sizeof( NodeValuePtr ) );
	if( buildNodeImage )
		memset( &node_image_[0], 0, w*h * sizeof( NodePtr ) );
	imageNodeAllocator_.reset();

	parallelInfoList.clear();

}


template <typename T>
MultiResolutionColorSurfelMap<T>::MultiResolutionColorSurfelMap( float minResolution, float maxRange, boost::shared_ptr< spatialaggregate::OcTreeNodeAllocator< float, T > > allocator ) {

	min_resolution_ = minResolution;
	max_range_ = maxRange;

	last_pair_surfel_idx_ = 0;

	reference_pose_.setIdentity();

	Eigen::Matrix< float, 4, 1 > center( 0.f, 0.f, 0.f, 0.f );
	allocator_ = allocator;
	octree_ = boost::shared_ptr< spatialaggregate::OcTree< float, T > >( new spatialaggregate::OcTree< float, T >( center, minResolution, maxRange, allocator ) );

	if ( !r ) {
		const gsl_rng_type* J = gsl_rng_default;
		gsl_rng_env_setup();
		r = gsl_rng_alloc( J );
	}


}

template <typename T>
MultiResolutionColorSurfelMap<T>::~MultiResolutionColorSurfelMap() {
}


template <typename T>
void MultiResolutionColorSurfelMap<T>::extents( Eigen::Matrix< double, 3, 1 >& mean, Eigen::Matrix< double, 3, 3 >& cov ) {

	std::list< spatialaggregate::OcTreeNode< float, T >* > nodes;
	octree_->root_->getAllLeaves( nodes );

	Eigen::Matrix< double, 3, 1 > sum;
	Eigen::Matrix< double, 3, 3 > sumSquares;
	double numPoints = 0;
	sum.setZero();
	sumSquares.setZero();

	for( typename std::list< spatialaggregate::OcTreeNode< float, T >* >::iterator it = nodes.begin(); it != nodes.end(); ++it ) {

		  T& v = (*it)->value_;

		  for( int i = 0; i < v.numberOfSurfels; i++ ) {

				Eigen::Vector3d mean_s = v.surfels_[i].mean_.template block<3,1>(0,0);
				double num_points_s = v.surfels_[i].num_points_;

				sum += num_points_s * mean_s;
				sumSquares += num_points_s * (v.surfels_[i].cov_.template block<3,3>(0,0) + mean_s * mean_s.transpose());
				numPoints += num_points_s;

		  }

	}

	if( numPoints > 0 ) {

		  const double inv_num = 1.0 / numPoints;
		  mean = sum * inv_num;
		  cov = inv_num * sumSquares - mean * mean.transpose();

	}

}

template <typename T>
void MultiResolutionColorSurfelMap<T>::addPoints( const boost::shared_ptr< const pcl::PointCloud< pcl::PointXYZRGB > >& cloud, const boost::shared_ptr< const std::vector< int > >& indices ) {
	addPoints( *cloud, *indices );
}

template <typename T>
void MultiResolutionColorSurfelMap<T>::addPoints( const pcl::PointCloud< pcl::PointXYZRGB >& cloud, const std::vector< int >& indices ) {

	Eigen::Vector3d sensorOrigin;
	for ( int i = 0; i < 3; i++ )
		sensorOrigin( i ) = cloud.sensor_origin_( i );

	const double inv_255 = 1.0 / 255.0;
	const float sqrt305 = 0.5f*sqrtf(3.f);
	const double max_dist = MAX_VIEWDIR_DIST;


	// go through the point cloud and add point information to map
	for ( unsigned int i = 0; i < indices.size(); i++ ) {

		const pcl::PointXYZRGB& p = cloud.points[indices[i]];
		const float x = p.x;
		const float y = p.y;
		const float z = p.z;

		if ( isnan( x ) || isinf( x ) )
			continue;

		if ( isnan( y ) || isinf( y ) )
			continue;

		if ( isnan( z ) || isinf( z ) )
			continue;

		float rgbf = p.rgb;
		unsigned int rgb = * ( reinterpret_cast< unsigned int* > ( &rgbf ) );
		unsigned int r = ( ( rgb & 0x00FF0000 ) >> 16 );
		unsigned int g = ( ( rgb & 0x0000FF00 ) >> 8 );
		unsigned int b = ( rgb & 0x000000FF );

		// HSL by Luminance and Cartesian Hue-Saturation (L-alpha-beta)
		float rf = inv_255*r, gf = inv_255*g, bf = inv_255*b;


		// RGB to L-alpha-beta:
		float L = 0.5f * ( std::max( std::max( rf, gf ), bf ) + std::min( std::min( rf, gf ), bf ) );
		float alpha = 0.5f * ( 2.f*rf - gf - bf );
		float beta = sqrt305 * (gf-bf);

		Eigen::Matrix< double, 6, 1 > pos;
		pos( 0 ) = p.x;
		pos( 1 ) = p.y;
		pos( 2 ) = p.z;
		pos( 3 ) = L;
		pos( 4 ) = alpha;
		pos( 5 ) = beta;


		Eigen::Vector3d viewDirection = pos.block< 3, 1 > ( 0, 0 ) - sensorOrigin;
		const double viewDistance = viewDirection.norm();

		if ( viewDistance < 1e-10 )
			continue;

		double viewDistanceInv = 1.0 / viewDistance;
		viewDirection *= viewDistanceInv;

		double distanceWeight = 1.0;

		MultiResolutionColorSurfelMap<T>::Surfel surfel;
		surfel.add( pos );
//		surfel.add( distanceWeight * pos, ( distanceWeight * pos ) * pos.transpose(), distanceWeight );
		surfel.first_view_dir_ = viewDirection;
		surfel.first_view_inv_dist_ = viewDistanceInv;

		T value;

		// add surfel to view directions within an angular interval
		for( unsigned int k = 0; k < value.numberOfSurfels; k++ ) {
			const double dist = viewDirection.dot( value.surfels_[k].initial_view_dir_ );
			if( dist > max_dist ) {
				value.surfels_[k] += surfel;
			}
		}


		// max resolution depends on depth: the farer, the bigger the minimumVolumeSize
		// see: http://www.ros.org/wiki/openni_kinect/kinect_accuracy
		// i roughly used the 90% percentile function for a single kinect
		int depth = ceil( octree_->depthForVolumeSize( std::max( (float) min_resolution_, (float) ( 2.f * params_.dist_dependency * viewDistance * viewDistance ) ) ) );

		spatialaggregate::OcTreeNode< float, T >* n = octree_->addPoint( p.getVector4fMap(), value, depth );


	}

}

template <typename T>
void MultiResolutionColorSurfelMap<T>::addImage( const pcl::PointCloud< pcl::PointXYZRGB >& cloud, bool smoothViewDir, bool buildNodeImage ) {


	imageAllocator_->prepare( cloud.width, cloud.height, buildNodeImage );
	int imageAggListIdx = 0;

	int idx = 0;
	const unsigned int width4 = 4*cloud.width;
	uint64_t* imgPtr = &imageAllocator_->imgKeys[0];
	T** mapPtr = &imageAllocator_->valueMap[0];

	const T initValue;

	Eigen::Vector4d sensorOrigin = cloud.sensor_origin_.cast<double>();
	const double sox = sensorOrigin(0);
	const double soy = sensorOrigin(1);
	const double soz = sensorOrigin(2);

	Eigen::Matrix4f sensorTransform = Eigen::Matrix4f::Identity();
	sensorTransform.block<4,1>(0,3) = cloud.sensor_origin_;
	sensorTransform.block<3,3>(0,0) = cloud.sensor_orientation_.matrix();


	const float inv_255 = 1.0 / 255.0;
	const float sqrt305 = 0.5f*sqrtf(3.f);
	const double max_dist = MAX_VIEWDIR_DIST;

	stopwatch_.reset();

	const float minpx = octree_->min_position_(0);
	const float minpy = octree_->min_position_(1);
	const float minpz = octree_->min_position_(2);

	const float pnx = octree_->position_normalizer_(0);
	const float pny = octree_->position_normalizer_(1);
	const float pnz = octree_->position_normalizer_(2);

	const int maxdepth = octree_->max_depth_;

	const int w = cloud.width;
	const int wm1 = w-1;
	const int wp1 = w+1;
	const int h = cloud.height;

	unsigned char depth = maxdepth;
	float minvolsize = octree_->minVolumeSizeForDepth( maxdepth );
	float maxvolsize = octree_->maxVolumeSizeForDepth( maxdepth );

	Eigen::Matrix< double, 6, 1 > pos;
	Eigen::Matrix< double, 1, 6 > posT;

	const float DIST_DEPENDENCY = params_.dist_dependency;

	for( int y = 0; y < h; y++ ) {

		uint64_t keyleft = 0;

		for( int x = 0; x < w; x++ ) {

			const pcl::PointXYZRGB& p = cloud.points[idx++];

			if( std::isnan( p.x ) ) {
				mapPtr++;
				imgPtr++;
				continue;
			}

			Eigen::Vector3d viewDirection( p.x - sox, p.y - soy, p.z - soz );
			const double viewDistance = viewDirection.norm();

			const float distdep = (2. * DIST_DEPENDENCY * viewDistance*viewDistance);

			const unsigned int kx_ = (p.x - minpx) * pnx + 0.5;
			const unsigned int ky_ = (p.y - minpy) * pny + 0.5;
			const unsigned int kz_ = (p.z - minpz) * pnz + 0.5;



			// try to avoid the log
			if( distdep < minvolsize || distdep > maxvolsize ) {

				depth = octree_->depthForVolumeSize( (double)distdep ) + 0.5f;

				if( depth >= maxdepth ) {
					depth = maxdepth;
				}

				minvolsize = octree_->minVolumeSizeForDepth( depth );
				maxvolsize = octree_->maxVolumeSizeForDepth( depth );

			}


			const unsigned int x_ = (kx_ >> (MAX_REPRESENTABLE_DEPTH-depth));
			const unsigned int y_ = (ky_ >> (MAX_REPRESENTABLE_DEPTH-depth));
			const unsigned int z_ = (kz_ >> (MAX_REPRESENTABLE_DEPTH-depth));

			uint64_t imgkey = (((uint64_t)x_ & 0xFFFFLL) << 48) | (((uint64_t)y_ & 0xFFFFLL) << 32) | (((uint64_t)z_ & 0xFFFFLL) << 16) | (uint64_t)depth;

			// check pixel above
			if( y > 0 ) {

				if( imgkey == *(imgPtr-w) )
					*mapPtr = *(mapPtr-w);
				else {

					if( imgkey == *(imgPtr-wp1) ) {
						*mapPtr = *(mapPtr-wp1);
					}
					else {

						// check pixel right
						if( x < wm1 ) {

							if( imgkey == *(imgPtr-wm1) ) {
								*mapPtr = *(mapPtr-wm1);
							}

						}

					}

				}

			}

			// check pixel before
			if( !*mapPtr && imgkey == keyleft ) {
				*mapPtr = *(mapPtr-1);
			}


			const double viewDistanceInv = 1.0 / viewDistance;
			viewDirection *= viewDistanceInv;

			if( !*mapPtr ) {
				// create new node value
				*mapPtr = imageAllocator_->imageNodeAllocator_.allocate();
				//memcpy( (*mapPtr)->surfels_, initValue.surfels_, sizeof(initValue.surfels_) );
				std::copy(initValue.surfels_.begin(), initValue.surfels_.end(), (*mapPtr)->surfels_.begin());
				for( unsigned int i = 0; i < (*mapPtr)->numberOfSurfels; i++ ) {
					(*mapPtr)->surfels_[i].first_view_dir_ = viewDirection;
					(*mapPtr)->surfels_[i].first_view_inv_dist_ = viewDistanceInv;
				}

				typename ImagePreAllocator::Info& info = imageAllocator_->infoList[imageAggListIdx];
				info.value = *mapPtr;
				info.key.x_ = kx_;
				info.key.y_ = ky_;
				info.key.z_ = kz_;
				info.depth = depth;

				imageAggListIdx++;

			}



			// add point to surfel
			const float rgbf = p.rgb;
			const unsigned int rgb = * ( reinterpret_cast< const unsigned int* > ( &rgbf ) );
			const unsigned int r = ( ( rgb & 0x00FF0000 ) >> 16 );
			const unsigned int g = ( ( rgb & 0x0000FF00 ) >> 8 );
			const unsigned int b = ( rgb & 0x000000FF );

			// HSL by Luminance and Cartesian Hue-Saturation (L-alpha-beta)
			const float rf = inv_255*(float)r;
			const float gf = inv_255*(float)g;
			const float bf = inv_255*(float)b;

			float maxch = rf;
			if( bf > maxch )
				maxch = bf;
			if( gf > maxch )
				maxch = gf;

			float minch = rf;
			if( bf < minch )
				minch = bf;
			if( gf < minch )
				minch = gf;

			const float L = 0.5f * ( maxch + minch );
			const float alpha = 0.5f * ( 2.f*rf - gf - bf );
			const float beta = sqrt305 * (gf-bf);


			pos( 0 ) = posT( 0 ) = p.x;
			pos( 1 ) = posT( 1 ) = p.y;
			pos( 2 ) = posT( 2 ) = p.z;
			pos( 3 ) = posT( 3 ) = L;
			pos( 4 ) = posT( 4 ) = alpha;
			pos( 5 ) = posT( 5 ) = beta;


			const Eigen::Matrix< double, 6, 6 > ppT = pos * posT;

			if( !smoothViewDir ) {
				MultiResolutionColorSurfelMap::Surfel* surfel = (*mapPtr)->getSurfel( viewDirection );
//				surfel->add( pos, ppT, 1.0 );
				surfel->add( pos );
			}
			else {
				// add surfel to view directions within an angular interval
				for( unsigned int k = 0; k < (*mapPtr)->numberOfSurfels; k++ ) {
					const double dist = viewDirection.dot( (*mapPtr)->surfels_[k].initial_view_dir_ );
					if( dist > max_dist ) {
//						(*mapPtr)->surfels_[k].add( pos, ppT, 1.0 );
						(*mapPtr)->surfels_[k].add( pos );
					}
				}
			}

			*imgPtr++ = keyleft = imgkey;
			mapPtr++;

		}
	}

	double delta_t = stopwatch_.getTimeSeconds() * 1000.0f;

//	std::cout << "aggregation took " << delta_t << "\n";

    stopwatch_.reset();

    for( unsigned int i = 0; i < imageAggListIdx; i++ ) {

    	const typename ImagePreAllocator::Info& info = imageAllocator_->infoList[i];
		spatialaggregate::OcTreeNode< float, T >* n = octree_->root_->addPoint( info.key, *info.value, info.depth );
		info.value->association_ = n;

    }

	delta_t = stopwatch_.getTimeSeconds() * 1000.0f;

//	std::cout << "tree construction took " << delta_t << "\n";

	imageAllocator_->node_set_.clear();

	if( buildNodeImage ) {

		T** mapPtr = &imageAllocator_->valueMap[0];
		unsigned int idx = 0;

		T* lastNodeValue = NULL;

		for( int y = 0; y < h; y++ ) {

			for( int x = 0; x < w; x++ ) {

				if( *mapPtr ) {
					imageAllocator_->node_image_[idx++] = (*mapPtr)->association_;
					if( *mapPtr != lastNodeValue ) {
						imageAllocator_->node_set_.insert( (*mapPtr)->association_ );
					}
				}
				else
					imageAllocator_->node_image_[idx++] = NULL;

				lastNodeValue = *mapPtr;
				mapPtr++;

			}
		}

	}

}

template <typename T>
void MultiResolutionColorSurfelMap<T>::addDisplacementImage( const pcl::PointCloud< pcl::PointXYZRGB >& cloud_pos, const pcl::PointCloud< pcl::PointXYZRGB >& cloud_disp, bool smoothViewDir, bool buildNodeImage ) {


	imageAllocator_->prepare( cloud_pos.width, cloud_pos.height, buildNodeImage );
	int imageAggListIdx = 0;

	int idx = 0;
	const unsigned int width4 = 4*cloud_pos.width;
	uint64_t* imgPtr = &imageAllocator_->imgKeys[0];
	T** mapPtr = &imageAllocator_->valueMap[0];

	const T initValue;

	Eigen::Vector4d sensorOrigin = cloud_pos.sensor_origin_.cast<double>();
	const double sox = sensorOrigin(0);
	const double soy = sensorOrigin(1);
	const double soz = sensorOrigin(2);

	Eigen::Matrix4f sensorTransform = Eigen::Matrix4f::Identity();
	sensorTransform.block<4,1>(0,3) = cloud_pos.sensor_origin_;
	sensorTransform.block<3,3>(0,0) = cloud_pos.sensor_orientation_.matrix();


	const float inv_255 = 1.0 / 255.0;
	const float sqrt305 = 0.5f*sqrtf(3.f);
	const double max_dist = MAX_VIEWDIR_DIST;

	stopwatch_.reset();

	const float minpx = octree_->min_position_(0);
	const float minpy = octree_->min_position_(1);
	const float minpz = octree_->min_position_(2);

	const float pnx = octree_->position_normalizer_(0);
	const float pny = octree_->position_normalizer_(1);
	const float pnz = octree_->position_normalizer_(2);

	const int maxdepth = octree_->max_depth_;

	const int w = cloud_pos.width;
	const int wm1 = w-1;
	const int wp1 = w+1;
	const int h = cloud_pos.height;

	unsigned char depth = maxdepth;
	float minvolsize = octree_->minVolumeSizeForDepth( maxdepth );
	float maxvolsize = octree_->maxVolumeSizeForDepth( maxdepth );

	Eigen::Matrix< double, 6, 1 > pos;
	Eigen::Matrix< double, 1, 6 > posT;

	const float DIST_DEPENDENCY = params_.dist_dependency;

	for( int y = 0; y < h; y++ ) {

		uint64_t keyleft = 0;

		for( int x = 0; x < w; x++ ) {

			const pcl::PointXYZRGB& p_disp = cloud_disp.points[idx];
			const pcl::PointXYZRGB& p = cloud_pos.points[idx++];

			if( std::isnan( p.x ) ) {
				mapPtr++;
				imgPtr++;
				continue;
			}

			Eigen::Vector3d viewDirection( p.x - sox, p.y - soy, p.z - soz );
			const double viewDistance = viewDirection.norm();

			const float distdep = (2. * DIST_DEPENDENCY * viewDistance*viewDistance);

			const unsigned int kx_ = (p.x - minpx) * pnx + 0.5;
			const unsigned int ky_ = (p.y - minpy) * pny + 0.5;
			const unsigned int kz_ = (p.z - minpz) * pnz + 0.5;



			// try to avoid the log
			if( distdep < minvolsize || distdep > maxvolsize ) {

				depth = octree_->depthForVolumeSize( (double)distdep ) + 0.5f;

				if( depth >= maxdepth ) {
					depth = maxdepth;
				}

				minvolsize = octree_->minVolumeSizeForDepth( depth );
				maxvolsize = octree_->maxVolumeSizeForDepth( depth );

			}


			const unsigned int x_ = (kx_ >> (MAX_REPRESENTABLE_DEPTH-depth));
			const unsigned int y_ = (ky_ >> (MAX_REPRESENTABLE_DEPTH-depth));
			const unsigned int z_ = (kz_ >> (MAX_REPRESENTABLE_DEPTH-depth));

			uint64_t imgkey = (((uint64_t)x_ & 0xFFFFLL) << 48) | (((uint64_t)y_ & 0xFFFFLL) << 32) | (((uint64_t)z_ & 0xFFFFLL) << 16) | (uint64_t)depth;

			// check pixel above
			if( y > 0 ) {

				if( imgkey == *(imgPtr-w) )
					*mapPtr = *(mapPtr-w);
				else {

					if( imgkey == *(imgPtr-wp1) ) {
						*mapPtr = *(mapPtr-wp1);
					}
					else {

						// check pixel right
						if( x < wm1 ) {

							if( imgkey == *(imgPtr-wm1) ) {
								*mapPtr = *(mapPtr-wm1);
							}

						}

					}

				}

			}

			// check pixel before
			if( !*mapPtr && imgkey == keyleft ) {
				*mapPtr = *(mapPtr-1);
			}


			const double viewDistanceInv = 1.0 / viewDistance;
			viewDirection *= viewDistanceInv;

			if( !*mapPtr ) {
				// create new node value
				*mapPtr = imageAllocator_->imageNodeAllocator_.allocate();
				//memcpy( (*mapPtr)->surfels_, initValue.surfels_, sizeof(initValue.surfels_) );
				std::copy(initValue.surfels_.begin(), initValue.surfels_.end(), (*mapPtr)->surfels_.begin());
				for( unsigned int i = 0; i < (*mapPtr)->numberOfSurfels; i++ ) {
					(*mapPtr)->surfels_[i].first_view_dir_ = viewDirection;
					(*mapPtr)->surfels_[i].first_view_inv_dist_ = viewDistanceInv;
				}

				typename ImagePreAllocator::Info& info = imageAllocator_->infoList[imageAggListIdx];
				info.value = *mapPtr;
				info.key.x_ = kx_;
				info.key.y_ = ky_;
				info.key.z_ = kz_;
				info.depth = depth;

				imageAggListIdx++;

			}



			// add point to surfel

			pos( 0 ) = posT( 0 ) = p.x;
			pos( 1 ) = posT( 1 ) = p.y;
			pos( 2 ) = posT( 2 ) = p.z;
			pos( 3 ) = posT( 3 ) = p_disp.x;
			pos( 4 ) = posT( 4 ) = p_disp.y;
			pos( 5 ) = posT( 5 ) = p_disp.z;


			const Eigen::Matrix< double, 6, 6 > ppT = pos * posT;

			if( !smoothViewDir ) {
				MultiResolutionColorSurfelMap::Surfel* surfel = (*mapPtr)->getSurfel( viewDirection );
//				surfel->add( pos, ppT, 1.0 );
				surfel->add( pos );
			}
			else {
				// add surfel to view directions within an angular interval
				for( unsigned int k = 0; k < (*mapPtr)->numberOfSurfels; k++ ) {
					const double dist = viewDirection.dot( (*mapPtr)->surfels_[k].initial_view_dir_ );
					if( dist > max_dist ) {
//						(*mapPtr)->surfels_[k].add( pos, ppT, 1.0 );
						(*mapPtr)->surfels_[k].add( pos );
					}
				}
			}

			*imgPtr++ = keyleft = imgkey;
			mapPtr++;

		}
	}

	double delta_t = stopwatch_.getTimeSeconds() * 1000.0f;

//	std::cout << "aggregation took " << delta_t << "\n";

    stopwatch_.reset();

    for( unsigned int i = 0; i < imageAggListIdx; i++ ) {

    	const typename ImagePreAllocator::Info& info = imageAllocator_->infoList[i];
		spatialaggregate::OcTreeNode< float, T >* n = octree_->root_->addPoint( info.key, *info.value, info.depth );
		info.value->association_ = n;

    }

	delta_t = stopwatch_.getTimeSeconds() * 1000.0f;

//	std::cout << "tree construction took " << delta_t << "\n";

	imageAllocator_->node_set_.clear();

	if( buildNodeImage ) {

		T** mapPtr = &imageAllocator_->valueMap[0];
		unsigned int idx = 0;

		T* lastNodeValue = NULL;

		for( int y = 0; y < h; y++ ) {

			for( int x = 0; x < w; x++ ) {

				if( *mapPtr ) {
					imageAllocator_->node_image_[idx++] = (*mapPtr)->association_;
					if( *mapPtr != lastNodeValue ) {
						imageAllocator_->node_set_.insert( (*mapPtr)->association_ );
					}
				}
				else
					imageAllocator_->node_image_[idx++] = NULL;

				lastNodeValue = *mapPtr;
				mapPtr++;

			}
		}

	}

}


template <typename T>
void MultiResolutionColorSurfelMap<T>::addImageParallel( const pcl::PointCloud< pcl::PointXYZRGB >& cloud, bool smoothViewDir, bool buildNodeImage ) {

    stopwatch_.reset();

    ImageAddFunctor af( this, cloud, smoothViewDir, buildNodeImage, imageAllocator_ );

	tbb::parallel_for( tbb::blocked_range<size_t>(0,cloud.height), af );

	double delta_t = stopwatch_.getTimeSeconds() * 1000.0f;

	std::cout << "aggregation took " << delta_t << "\n";


    stopwatch_.reset();

    for( typename tbb::concurrent_vector< std::vector< typename ImagePreAllocator::Info > >::iterator it = imageAllocator_->parallelInfoList.begin(); it != imageAllocator_->parallelInfoList.end(); ++it ) {

    	for( typename std::vector< typename ImagePreAllocator::Info >::iterator it2 = it->begin(); it2 != it->end(); ++it2 ) {
			const typename ImagePreAllocator::Info& info = *it2;
			if( !info.value )
				break;
			spatialaggregate::OcTreeNode< float, T >* n = octree_->root_->addPoint( info.key, *info.value, info.depth );
			info.value->association_ = n;
    	}

    }

	delta_t = stopwatch_.getTimeSeconds() * 1000.0f;

	std::cout << "tree construction took " << delta_t << "\n";

	imageAllocator_->node_set_.clear();

	if( buildNodeImage ) {

		const int w = cloud.width;
		const int h = cloud.height;

		T** mapPtr = &imageAllocator_->valueMap[0];
		unsigned int idx = 0;

		T* lastNodeValue = NULL;

		for( int y = 0; y < h; y++ ) {

			for( int x = 0; x < w; x++ ) {

				if( *mapPtr ) {
					imageAllocator_->node_image_[idx++] = (*mapPtr)->association_;
					if( *mapPtr != lastNodeValue ) {
						imageAllocator_->node_set_.insert( (*mapPtr)->association_ );
					}
				}
				else
					imageAllocator_->node_image_[idx++] = NULL;

				lastNodeValue = *mapPtr;
				mapPtr++;

			}
		}

	}

}

template <typename T>
MultiResolutionColorSurfelMap<T>::ImageAddFunctor::ImageAddFunctor( MultiResolutionColorSurfelMap* map, const pcl::PointCloud< pcl::PointXYZRGB >& cloud, bool smoothViewDir, bool buildNodeImage, boost::shared_ptr< ImagePreAllocator > imageAllocator )
: map_( map ), cloud_( cloud )  {

	smoothViewDir_ = smoothViewDir;
	imageAllocator_ = imageAllocator;
	imageAllocator_->prepare( cloud.width, cloud.height, buildNodeImage );

}


template <typename T>
void MultiResolutionColorSurfelMap<T>::ImageAddFunctor::operator()( const tbb::blocked_range<size_t>& r ) const {

	int idx = r.begin()*cloud_.width;

//	tbb::concurrent_vector< std::vector< ImagePreAllocator::Info >::iterator iit =
	std::vector< typename ImagePreAllocator::Info >& infoList = *(imageAllocator_->parallelInfoList.push_back( std::vector< typename ImagePreAllocator::Info >() ));
	infoList.resize( cloud_.width * (r.end()-r.begin()) );
	unsigned int infoListIdx = 0;

	const unsigned int width4 = 4*cloud_.width;
	uint64_t* imgPtr = &imageAllocator_->imgKeys[idx];
	T** mapPtr = &imageAllocator_->valueMap[idx];

	const T initValue;

	Eigen::Vector4d sensorOrigin = cloud_.sensor_origin_.cast<double>();
	const double sox = sensorOrigin(0);
	const double soy = sensorOrigin(1);
	const double soz = sensorOrigin(2);

	Eigen::Matrix4f sensorTransform = Eigen::Matrix4f::Identity();
	sensorTransform.block<4,1>(0,3) = cloud_.sensor_origin_;
	sensorTransform.block<3,3>(0,0) = cloud_.sensor_orientation_.matrix();

	const float inv_255 = 1.0 / 255.0;
	const float sqrt305 = 0.5f*sqrtf(3.f);
	const double max_dist = MAX_VIEWDIR_DIST;

	const float minpx = map_->octree_->min_position_(0);
	const float minpy = map_->octree_->min_position_(1);
	const float minpz = map_->octree_->min_position_(2);

	const float pnx = map_->octree_->position_normalizer_(0);
	const float pny = map_->octree_->position_normalizer_(1);
	const float pnz = map_->octree_->position_normalizer_(2);

	const int maxdepth = map_->octree_->max_depth_;

	const int w = cloud_.width;
	const int wm1 = w-1;
	const int wp1 = w+1;
	const int h = cloud_.height;

	unsigned char depth = maxdepth;
	float minvolsize = map_->octree_->minVolumeSizeForDepth( maxdepth );
	float maxvolsize = map_->octree_->maxVolumeSizeForDepth( maxdepth );

	Eigen::Matrix< double, 6, 1 > pos;
	Eigen::Matrix< double, 1, 6 > posT;

	const float DIST_DEPENDENCY = map_->params_.dist_dependency;

	for( int y = r.begin(); y < r.end(); y++ ) {

		uint64_t keyleft = 0;

		for( int x = 0; x < w; x++ ) {

			const pcl::PointXYZRGB& p = cloud_.points[idx++];

			if( std::isnan( p.x ) ) {
				mapPtr++;
				imgPtr++;
				continue;
			}

			Eigen::Vector3d viewDirection( p.x - sox, p.y - soy, p.z - soz );
			const double viewDistance = viewDirection.norm();

			const float distdep = (2. * DIST_DEPENDENCY * viewDistance*viewDistance);

			const unsigned int kx_ = (p.x - minpx) * pnx + 0.5;
			const unsigned int ky_ = (p.y - minpy) * pny + 0.5;
			const unsigned int kz_ = (p.z - minpz) * pnz + 0.5;



			// try to avoid the log
			if( distdep < minvolsize || distdep > maxvolsize ) {

				depth = map_->octree_->depthForVolumeSize( (double)distdep ) + 0.5f;

				if( depth >= maxdepth ) {
					depth = maxdepth;
				}

				minvolsize = map_->octree_->minVolumeSizeForDepth( depth );
				maxvolsize = map_->octree_->maxVolumeSizeForDepth( depth );

			}


			const unsigned int x_ = (kx_ >> (MAX_REPRESENTABLE_DEPTH-depth));
			const unsigned int y_ = (ky_ >> (MAX_REPRESENTABLE_DEPTH-depth));
			const unsigned int z_ = (kz_ >> (MAX_REPRESENTABLE_DEPTH-depth));

			uint64_t imgkey = (((uint64_t)x_ & 0xFFFFLL) << 48) | (((uint64_t)y_ & 0xFFFFLL) << 32) | (((uint64_t)z_ & 0xFFFFLL) << 16) | (uint64_t)depth;

			// check pixel above
			// dont access beyond our blocked range
			if( y > r.begin() ) {

				if( imgkey == *(imgPtr-w) )
					*mapPtr = *(mapPtr-w);
				else {

					if( imgkey == *(imgPtr-wp1) ) {
						*mapPtr = *(mapPtr-wp1);
					}
					else {

						// check pixel right
						if( x < wm1 ) {

							if( imgkey == *(imgPtr-wm1) ) {
								*mapPtr = *(mapPtr-wm1);
							}

						}

					}

				}

			}

			// check pixel before
			if( !*mapPtr && imgkey == keyleft ) {
				*mapPtr = *(mapPtr-1);
			}


			const double viewDistanceInv = 1.0 / viewDistance;
			viewDirection *= viewDistanceInv;

			if( !*mapPtr ) {
				// create new node value
				*mapPtr = imageAllocator_->imageNodeAllocator_.concurrent_allocate();
				//memcpy( (*mapPtr)->surfels_, initValue.surfels_, sizeof(initValue.surfels_) );
				std::copy(initValue.surfels_.begin(), initValue.surfels_.end(), (*mapPtr)->surfels_.begin());

				for( unsigned int i = 0; i < (*mapPtr)->numberOfSurfels; i++ ) {
					(*mapPtr)->surfels_[i].first_view_dir_ = viewDirection;
					(*mapPtr)->surfels_[i].first_view_inv_dist_ = viewDistanceInv;
				}

				typename ImagePreAllocator::Info& info = infoList[infoListIdx];
				info.value = *mapPtr;
				info.key.x_ = kx_;
				info.key.y_ = ky_;
				info.key.z_ = kz_;
				info.depth = depth;

				infoListIdx++;

			}



			// add point to surfel
			const float rgbf = p.rgb;
			const unsigned int rgb = * ( reinterpret_cast< const unsigned int* > ( &rgbf ) );
			const unsigned int r = ( ( rgb & 0x00FF0000 ) >> 16 );
			const unsigned int g = ( ( rgb & 0x0000FF00 ) >> 8 );
			const unsigned int b = ( rgb & 0x000000FF );

			// HSL by Luminance and Cartesian Hue-Saturation (L-alpha-beta)
			const float rf = inv_255*(float)r;
			const float gf = inv_255*(float)g;
			const float bf = inv_255*(float)b;

			float maxch = rf;
			if( bf > maxch )
				maxch = bf;
			if( gf > maxch )
				maxch = gf;

			float minch = rf;
			if( bf < minch )
				minch = bf;
			if( gf < minch )
				minch = gf;

			const float L = 0.5f * ( maxch + minch );
			const float alpha = 0.5f * ( 2.f*rf - gf - bf );
			const float beta = sqrt305 * (gf-bf);


			pos( 0 ) = posT( 0 ) = p.x;
			pos( 1 ) = posT( 1 ) = p.y;
			pos( 2 ) = posT( 2 ) = p.z;
			pos( 3 ) = posT( 3 ) = L;
			pos( 4 ) = posT( 4 ) = alpha;
			pos( 5 ) = posT( 5 ) = beta;


			const Eigen::Matrix< double, 6, 6 > ppT = pos * posT;

			if( !smoothViewDir_ ) {
				MultiResolutionColorSurfelMap::Surfel* surfel = (*mapPtr)->getSurfel( viewDirection );
//				surfel->add( pos, ppT, 1.0 );
				surfel->add( pos );
			}
			else {
				// add surfel to view directions within an angular interval
				for( unsigned int k = 0; k < (*mapPtr)->numberOfSurfels; k++ ) {
					const double dist = viewDirection.dot( (*mapPtr)->surfels_[k].initial_view_dir_ );
					if( dist > max_dist ) {
//						(*mapPtr)->surfels_[k].add( pos, ppT, 1.0 );
						(*mapPtr)->surfels_[k].add( pos );
					}
				}
			}

			*imgPtr++ = keyleft = imgkey;
			mapPtr++;

		}
	}

}


struct KeypointComparator {
	bool operator() ( unsigned int i, unsigned int j ) {
		return (*keypoints_)[i].response > (*keypoints_)[j].response;
	}

	std::vector< cv::KeyPoint >* keypoints_;
};


// requires cloud in sensor frame
template <typename T>
void MultiResolutionColorSurfelMap<T>::addImagePointFeatures( const cv::Mat& img, const pcl::PointCloud< pcl::PointXYZRGB >& cloud ) {

	img_rgb_ = img;

	lsh_index_.reset();
	features_.clear();
	descriptors_ = cv::Mat();

	const float DIST_DEPENDENCY = params_.dist_dependency;

	const float pixelNoise = params_.pixelNoise;
	const float pixelNoise2 = pixelNoise*pixelNoise;
	const float depthNoiseScale2 = params_.depthNoiseFactor * DIST_DEPENDENCY * DIST_DEPENDENCY;

//	const float pixelNoiseAssoc = params_.pixelNoiseAssoc;//20.f;
//	const float pixelNoiseAssoc2 = pixelNoiseAssoc*pixelNoiseAssoc;
//	const float depthNoiseAssocScale2 = params_.depthNoiseAssocFactor * DIST_DEPENDENCY * DIST_DEPENDENCY; //4*4

	const float imgScaleFactor = (float)img.cols / 640.f;
	const int imageSearchRadius = 50 * imgScaleFactor;
	const int imageSearchRadius2 = imageSearchRadius*imageSearchRadius;
	const int descriptorDissimilarityThreshold = 30;
	const int descriptorSimilarityThreshold = 60;

	const int height = img.rows;
	const int width = img.cols;
	const int depthWindowSize = 2;

	float inv_focallength = 1.f / 525.f / imgScaleFactor;
	const float centerX = 0.5f * width * imgScaleFactor - 0.5f;
	const float centerY = 0.5f * height * imgScaleFactor - 0.5f;

	Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
	transform.block<3,3>(0,0) = Eigen::Matrix3d( cloud.sensor_orientation_.cast<double>() );
	transform.block<4,1>(0,3) = cloud.sensor_origin_.cast<double>();

//	std::cout << transform << "\n";

	Eigen::Matrix4d jac = Eigen::Matrix4d::Identity();
	Eigen::Matrix4d rot = Eigen::Matrix4d::Identity();
	rot.block<3,3>(0,0) = transform.block<3,3>(0,0);



    stopwatch_.reset();

	double delta_t;

	// extract ORB features (OpenCV 2.4.6)

	//CV_WRAP explicit ORB(int nfeatures = 500, float scaleFactor = 1.2f, int nlevels = 8, int edgeThreshold = 31,
    //int firstLevel = 0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31 );

	cv::ORB orb( params_.numPointFeatures, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31 );

	const Eigen::Vector3d so = transform.block<3,1>(0,3);
	const Eigen::Quaterniond sori( transform.block<3,3>(0,0) );

	static std::vector< cv::KeyPoint > last_keypoints;
	static cv::Mat last_descriptors;
	static cv::Mat last_img;

	// in bytes
	unsigned int descriptorSize = orb.descriptorSize();

	std::vector< cv::KeyPoint > detectedKeypoints, keypoints;
	detectedKeypoints.reserve( params_.numPointFeatures );

	orb.detect( img, detectedKeypoints, cv::Mat() );

	std::cout << "detect: " << stopwatch_.getTimeSeconds() * 1000.0 << "\n";
	stopwatch_.reset();

	// bin detections in grid and restrict to specific number of detections per bin
	const size_t MaxFeaturesPerGridCell = params_.GridCellMax;
	const unsigned int rows = params_.GridRows;
	const unsigned int cols = params_.GridCols;
	const unsigned int colWidth = width / cols;
	const unsigned int rowHeight = height / rows;

	std::vector< unsigned int > keypointGrid[ params_.GridRows ][ params_.GridCols ];

	for( unsigned int y = 0; y < params_.GridRows; y++ )
		for( unsigned int x = 0; x < params_.GridCols; x++ )
			keypointGrid[y][x].reserve( 100 );

	for( unsigned int i = 0; i < detectedKeypoints.size(); i++ ) {
		unsigned int gridx = std::max( 0.f, std::min( (float)params_.GridCols, detectedKeypoints[i].pt.x / colWidth ) );
		unsigned int gridy = std::max( 0.f, std::min( (float)params_.GridRows, detectedKeypoints[i].pt.y / rowHeight ) );
		keypointGrid[gridy][gridx].push_back( i );
	}


	KeypointComparator keypointComparator;
	keypointComparator.keypoints_ = &detectedKeypoints;

	std::vector< cv::KeyPoint > gridSampledKeypoints;
	gridSampledKeypoints.reserve( params_.numPointFeatures );
	for( unsigned int y = 0; y < params_.GridRows; y++ ) {
		for( unsigned int x = 0; x < params_.GridCols; x++ ) {
			// sort by response in descending order
			std::sort( keypointGrid[y][x].begin(), keypointGrid[y][x].end(), keypointComparator );
			// keep only specific number of strongest features
			keypointGrid[y][x].resize( std::min( keypointGrid[y][x].size(), (size_t) params_.GridCellMax ) );
			// add to new keypoint list
			for( unsigned int i = 0; i < keypointGrid[y][x].size(); i++ )
				gridSampledKeypoints.push_back( detectedKeypoints[keypointGrid[y][x][i]] );
		}
	}

	detectedKeypoints = gridSampledKeypoints;

	std::cout << "gridsample: " << stopwatch_.getTimeSeconds() * 1000.0 << "\n";
	stopwatch_.reset();

	cv::Mat detectedDescriptors;
	orb.compute( img, detectedKeypoints, detectedDescriptors );

	std::cout << "extract: " << stopwatch_.getTimeSeconds() * 1000.0 << "\n";




//	std::vector< std::vector< std::vector< doubleSort > > > imgGrid( rows );
//	for( int y = 0; y < rows; y++ )
//		imgGrid[ y ].resize( cols );
//
//	//add keypoints to imgGrid cell vectors
//	unsigned int NDetectedKeyPoints = detectedKeypoints.size();
//	for (unsigned int i = 0; i<NDetectedKeyPoints; i++)
//	{
//		cv::KeyPoint* kp = &(detectedKeypoints[i]);
//		doubleSort ds = { kp , i };
//
//		unsigned int colIndex = kp->pt.x / colWidth;
//		unsigned int rowIndex = kp->pt.y / rowHeight;
//		imgGrid[rowIndex][colIndex].push_back( ds );
//	}
//
//	// sort KeyPoints in grid cells
//	for (unsigned int row = 0; row < rows; row++)
//		for (unsigned int col = 0; col < cols; col++)
//			std::sort( imgGrid[row][col].begin(), imgGrid[row][col].end(), compareKeypoints );
//
//	// renew detectedKeypoints
//	cv::Mat detectedDescriptorsNew( detectedDescriptors.rows, detectedDescriptors.cols, CV_8UC1, 0.f );
//	std::vector< cv::KeyPoint > detectedKeypointsNew;
//	for (unsigned int row = 0; row < rows; row++)
//		for (unsigned int col = 0; col < cols; col++)
//		{
//			for (unsigned int keyNr = 0; keyNr < std::min(imgGrid[row][col].size(), MaxFeaturesPerGridCell ) ; keyNr++ )
//			{
//				int index = imgGrid[row][col][keyNr].index;
//				detectedKeypointsNew.push_back( detectedKeypoints[index] );
//
//				// Eintrag für Eintrag kopieren (geht nicht anders?)
//				for ( int descriptorElement = 0; descriptorElement < descriptorSize; descriptorElement++ )
//					detectedDescriptorsNew.data[ (detectedKeypointsNew.size()-1) * descriptorSize + descriptorElement ] = detectedDescriptors.data[ index * descriptorSize + descriptorElement ];
//			}
//		}
//	detectedDescriptors = detectedDescriptorsNew;
//	detectedKeypoints = detectedKeypointsNew;



	if( detectedKeypoints.size() < 3 ) {
		keypoints.clear();
	}
	else {

		cv::Mat tmpDescriptors = cv::Mat( features_.size() + detectedDescriptors.rows, detectedDescriptors.cols, CV_8UC1 );
		if( features_.size() > 0 )
			memcpy( &tmpDescriptors.data[ 0 ], &descriptors_.data[ 0 ], features_.size() * descriptorSize * sizeof( unsigned char ) );

		// build search index on image coordinates
		cv::Mat imageCoordinates = cv::Mat( (int)detectedKeypoints.size(), 2, CV_32FC1 ,0.f );
		for( size_t i = 0; i < detectedKeypoints.size(); i++ ) {
			imageCoordinates.at<float>( i, 0 ) = detectedKeypoints[i].pt.x;
			imageCoordinates.at<float>( i, 1 ) = detectedKeypoints[i].pt.y;
		}
		flann::Matrix< float > indexImageCoordinates( (float*)imageCoordinates.data, detectedKeypoints.size(), 2 );
		flann::Index< flann::L2_Simple< float > > image_index( indexImageCoordinates, flann::KDTreeSingleIndexParams() );
		image_index.buildIndex();

		std::vector< std::vector< int > > foundImageIndices;
		std::vector< std::vector< float > > foundImageDists;
		image_index.radiusSearch( indexImageCoordinates, foundImageIndices, foundImageDists, imageSearchRadius2, flann::SearchParams( 32, 0, false ) );

		flann::HammingPopcnt< unsigned char > hammingDist;
		keypoints.clear();
		keypoints.reserve( detectedKeypoints.size() );
		for( size_t i = 0; i < detectedKeypoints.size(); i++ ) {

			unsigned char* descriptor_i = &detectedDescriptors.data[ i * descriptorSize ];

			bool foundSimilarFeature = false;
			for( size_t j = 0; j < foundImageIndices[i].size(); j++ ) {

				size_t k = foundImageIndices[i][j];

				if( i == k )
					continue;

				if( k >= 0 && k < detectedKeypoints.size() ) {
					// compare descriptors.. results not sorted by descriptor similarity!
					const unsigned char* descriptor_k = &detectedDescriptors.data[ k * descriptorSize ];
//					if( hammingDist( descriptor_i, descriptor_k, descriptorSize ) < descriptorSimilarityThreshold ) {
					if( detectedKeypoints[i].response < detectedKeypoints[k].response && hammingDist( descriptor_i, descriptor_k, descriptorSize ) < descriptorDissimilarityThreshold ) {
//					if( detectedKeypoints[i].octave == detectedKeypoints[k].octave && hammingDist( descriptor_i, descriptor_k, descriptorSize ) < descriptorSimilarityThreshold ) {
						foundSimilarFeature = true;
						break;
					}
				}

			}

			if( !foundSimilarFeature ) {

				memcpy( &tmpDescriptors.data[ (features_.size() + keypoints.size()) * descriptorSize ], descriptor_i, descriptorSize * sizeof( unsigned char ) );
				keypoints.push_back( detectedKeypoints[ i ] );

			}
		}

		descriptors_ = tmpDescriptors.rowRange( 0, features_.size() + keypoints.size() ).clone();
	}

	std::cout << "sorted out " << detectedKeypoints.size()-keypoints.size() << " / " << detectedKeypoints.size() << " features for concurrent similarity in image position and descriptor\n";
	std::cout << "map has " << features_.size() + keypoints.size() << " features\n";




	// add features to the map
	size_t startFeatureIdx = features_.size();
	features_.resize( features_.size() + keypoints.size() );
	for( unsigned int i = 0; i < keypoints.size(); i++ ) {

		PointFeature& f = features_[startFeatureIdx+i];
		const cv::KeyPoint& kp = keypoints[i];

		// set inverse depth parametrization from point cloud
		bool hasDepth = false;
		double z = std::numeric_limits<double>::quiet_NaN();
		int minx = std::max( 0, (int)kp.pt.x - depthWindowSize );
		int maxx = std::min( width-1, (int)kp.pt.x + depthWindowSize );
		int miny = std::max( 0, (int)kp.pt.y - depthWindowSize );
		int maxy = std::min( height-1, (int)kp.pt.y + depthWindowSize );
		double sum_z = 0;
		double sum2_z = 0;
		double num_z = 0;
		for( int y = miny; y <= maxy; y++ ) {
			for( int x = minx; x <= maxx; x++ ) {

				int idx = y*width+x;
				const pcl::PointXYZRGB& p = cloud.points[idx];
				if( std::isnan( p.x ) ) {
					continue;
				}

				Eigen::Vector4d pos( p.x, p.y, p.z, 1.0 );
				pos = (transform.inverse() * pos).eval();

				sum_z += pos(2);
				sum2_z += pos(2)*pos(2);
				num_z += 1.f;

				if( std::isnan( z ) )
					z = pos(2);
				else
					z = std::min( z, pos(2) );
				hasDepth = true;

			}
		}

		// found depth?

		f.has_depth_ = hasDepth;

		if( hasDepth ) {

			float xi = inv_focallength * (kp.pt.x-centerX);
			float yi = inv_focallength * (kp.pt.y-centerY);

			f.image_pos_(0) = xi;
			f.image_pos_(1) = yi;

			f.pos_(0) = xi * z;
			f.pos_(1) = yi * z;
			f.pos_(2) = z;
			f.pos_(3) = 1.0;

            jac(0,0) = inv_focallength * z;
            jac(0,2) = xi;
            jac(1,1) = inv_focallength * z;
            jac(1,2) = yi;

			f.invzpos_(0) = kp.pt.x;
			f.invzpos_(1) = kp.pt.y;
			f.invzpos_(2) = 1.0 / z;
			f.invzpos_(3) = 1.0;


			// depth variance depends on depth..
			// propagate variance in depth to variance in inverse depth
			const double z4 = z*z*z*z;
			const double z_cov_emp = sum2_z / num_z - sum_z*sum_z / (num_z*num_z);

			f.cov_.setIdentity();
			f.cov_(0,0) = pixelNoise2; // in pixel^2
			f.cov_(1,1) = pixelNoise2; // in pixel^2
			f.cov_(2,2) = (depthNoiseScale2 * z4 + z_cov_emp);
            f.cov_ = (jac * f.cov_ * jac.transpose()).eval();

//			f.assoc_cov_.setIdentity();
//			f.assoc_cov_(0,0) = params_.pixelNoiseAssocFactor * pixelNoise2;
//			f.assoc_cov_(1,1) = params_.pixelNoiseAssocFactor * pixelNoise2;
//			f.assoc_cov_(2,2) = params_.depthNoiseAssocFactor * (depthNoiseScale2 * z4 + z_cov_emp);
//            f.assoc_cov_ = (jac * f.assoc_cov_ * jac.transpose()).eval();

			f.image_cov_.setIdentity();
			f.image_cov_ *= inv_focallength*inv_focallength*pixelNoise2;

//			f.image_assoc_cov_.setIdentity();
//			f.image_assoc_cov_ *= inv_focallength*inv_focallength*params_.pixelNoiseAssocFactor * pixelNoise2;



			f.invzcov_.setIdentity();
			f.invzcov_(0,0) = pixelNoise2; // in pixel^2
			f.invzcov_(1,1) = pixelNoise2; // in pixel^2
			f.invzcov_(2,2) = (depthNoiseScale2 * z4 + z_cov_emp) / z4;

//			f.assoc_invzcov_.setIdentity();
//			f.assoc_invzcov_(0,0) = params_.pixelNoiseAssocFactor * pixelNoise2;
//			f.assoc_invzcov_(1,1) = params_.pixelNoiseAssocFactor * pixelNoise2;
//			f.assoc_invzcov_(2,2) = params_.depthNoiseAssocFactor * (depthNoiseScale2 * z4 + z_cov_emp) / z4;



//			// add feature to corresponding surfels (at all resolutions)
//			spatialaggregate::OcTreeKey< float, T > poskey = octree_->getKey( mapPos(0), mapPos(1), mapPos(2) );
//
//			spatialaggregate::OcTreeNode< float, T >* node = octree_->root_;
//			while( node ) {
//
//				node->value_.addFeature( viewDirection, f );
//
//				node = node->children_[node->getOctant( poskey )];
//			}


		}
		else {

			// init to unknown depth
			z = 10.f;
			double z_cov_emp = z*z*0.25f;

			float xi = inv_focallength * (kp.pt.x-centerX);
			float yi = inv_focallength * (kp.pt.y-centerY);

			f.image_pos_(0) = xi;
			f.image_pos_(1) = yi;

			f.pos_(0) = xi * z;
			f.pos_(1) = yi * z;
			f.pos_(2) = z;
			f.pos_(3) = 1.0;

            jac(0,0) = inv_focallength * z;
            jac(0,2) = xi;
            jac(1,1) = inv_focallength * z;
            jac(1,2) = yi;


			f.invzpos_(0) = kp.pt.x;
			f.invzpos_(1) = kp.pt.y;
			f.invzpos_(2) = 1.0 / z;
			f.invzpos_(3) = 1.0;



			// covariance depends on depth..
			// propagate variance in depth to variance in inverse depth
			// add uncertainty from depth estimate in local image region
			const double z4 = z*z*z*z;

			f.cov_.setIdentity();
			f.cov_(0,0) = pixelNoise2; // in pixel^2
			f.cov_(1,1) = pixelNoise2; // in pixel^2
			f.cov_(2,2) = depthNoiseScale2 * z4 + z_cov_emp;
            f.cov_ = (jac * f.cov_ * jac.transpose()).eval();

//			f.assoc_cov_.setIdentity();
//			f.assoc_cov_(0,0) = params_.pixelNoiseAssocFactor * pixelNoise2;
//			f.assoc_cov_(1,1) = params_.pixelNoiseAssocFactor * pixelNoise2;
//			f.assoc_cov_(2,2) = params_.depthNoiseAssocFactor * (depthNoiseScale2 * z4 + z_cov_emp);
//            f.assoc_cov_ = (jac * f.assoc_cov_ * jac.transpose()).eval();

			f.image_cov_.setIdentity();
			f.image_cov_ *= inv_focallength*inv_focallength*pixelNoise2;

			f.image_assoc_cov_.setIdentity();
			f.image_assoc_cov_ *= inv_focallength*inv_focallength*params_.pixelNoiseAssocFactor * pixelNoise2;


			f.invzcov_.setIdentity();
			f.invzcov_(0,0) = pixelNoise2; // in pixel^2
			f.invzcov_(1,1) = pixelNoise2; // in pixel^2
			f.invzcov_(2,2) = depthNoiseScale2;


//			f.assoc_invzcov_.setIdentity();
//			f.assoc_invzcov_(0,0) = params_.pixelNoiseAssocFactor * pixelNoise2;
//			f.assoc_invzcov_(1,1) = params_.pixelNoiseAssocFactor * pixelNoise2;
//			f.assoc_invzcov_(2,2) = params_.depthNoiseAssocFactor * (depthNoiseScale2 * z4 + z_cov_emp) / z4;


			// features without depth can only be added to the root node, they are valid for all nodes and will be associated afterwards
//			octree_->root_->value_.addFeature( viewDirection, f );

		}

		f.pos_ = (transform * f.pos_).eval();
		f.cov_ = (rot * f.cov_ * rot.transpose()).eval();
//		f.assoc_cov_ = (rot * f.assoc_cov_ * rot.transpose()).eval();

		f.invzinvcov_ = f.invzcov_.inverse();

		f.origin_ = so;
		f.orientation_ = sori;



	}

	delta_t = stopwatch_.getTimeSeconds() * 1000.0f;
    std::cerr << "feature extraction took " << delta_t << "ms.\n";


	// build LSH search index for this image using LSH implementation in FLANN 1.7.1
    stopwatch_.reset();
    flann::Matrix< unsigned char > indexDescriptors( descriptors_.data, keypoints.size(), orb.descriptorSize() );
	lsh_index_ = boost::shared_ptr< flann::Index< flann::HammingPopcnt< unsigned char > > >( new flann::Index< flann::HammingPopcnt< unsigned char > >( indexDescriptors, flann::LshIndexParams( 2, 20, 2 ) ) );
	lsh_index_->buildIndex();
	delta_t = stopwatch_.getTimeSeconds() * 1000.0f;
    std::cerr << "lsh search index construction took " << delta_t << "ms.\n";




    if( params_.debugPointFeatures ) {
		cv::Mat outimg;
		cv::drawKeypoints( img, keypoints, outimg );
		cv::imshow( "keypoints", outimg );
		cv::waitKey(1);
    }




}


template <typename T>
void MultiResolutionColorSurfelMap<T>::getImage( cv::Mat& img, const Eigen::Vector3d& viewPosition ) {

	int h = imageAllocator_->height;
	int w = imageAllocator_->width;

	img = cv::Mat( h, w, CV_8UC3, 0.f );

	spatialaggregate::OcTreeNode< float, T >** nodeImgPtr = &imageAllocator_->node_image_[0];

	cv::Vec3b v;
	for( int y = 0; y < h; y++ ) {

		for( int x = 0; x < w; x++ ) {

			if( *nodeImgPtr ) {

				float rf = 0, gf = 0, bf = 0;
				Eigen::Vector3d viewDirection = (*nodeImgPtr)->getPosition().template block<3,1>(0,0).template cast<double>() - viewPosition;
				viewDirection.normalize();

				MultiResolutionColorSurfelMap::Surfel* surfel = (*nodeImgPtr)->value_.getSurfel( viewDirection );
				Eigen::Matrix< double, 6, 1 > vec = (*nodeImgPtr)->value_.getSurfel( viewDirection )->mean_;

				const float L = vec( 3 );
				const float alpha = vec( 4 );
				const float beta = vec( 5 );

				convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );

				int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
				int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
				int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );

				v[0] = b;
				v[1] = g;
				v[2] = r;

			}
			else {

				v[0] = 0;
				v[1] = 0;
				v[2] = 0;

			}

			img.at< cv::Vec3b >(y,x) = v;

			nodeImgPtr++;

		}
	}

}


template <typename T>
inline bool MultiResolutionColorSurfelMap<T>::splitCriterion( spatialaggregate::OcTreeNode< float, T >* oldLeaf, spatialaggregate::OcTreeNode< float, T >* newLeaf ) {

	return true;

}

template <typename T>
void MultiResolutionColorSurfelMap<T>::findImageBorderPoints( const pcl::PointCloud< pcl::PointXYZRGB >& cloud, std::vector< int >& indices ) {

	// determine first image points from the borders that are not nan

	// horizontally
	for ( unsigned int y = 0; y < cloud.height; y++ ) {

		for ( unsigned int x = 0; x < cloud.width; x++ ) {

			int idx = y * cloud.width + x;

			// if not nan, push back and break
			const pcl::PointXYZRGB& p = cloud.points[idx];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if ( isnan( px ) || isinf( px ) )
				continue;

			if ( isnan( py ) || isinf( py ) )
				continue;

			if ( isnan( pz ) || isinf( pz ) )
				continue;

			indices.push_back( idx );
			break;

		}

		for ( int x = cloud.width - 1; x >= 0; x-- ) {

			int idx = y * cloud.width + x;

			// if not nan, push back and break
			const pcl::PointXYZRGB& p = cloud.points[idx];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if ( isnan( px ) || isinf( px ) )
				continue;

			if ( isnan( py ) || isinf( py ) )
				continue;

			if ( isnan( pz ) || isinf( pz ) )
				continue;

			indices.push_back( idx );
			break;

		}

	}

	// vertically
	for ( unsigned int x = 0; x < cloud.width; x++ ) {

		for ( unsigned int y = 0; y < cloud.height; y++ ) {

			int idx = y * cloud.width + x;

			// if not nan, push back and break
			const pcl::PointXYZRGB& p = cloud.points[idx];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if ( isnan( px ) || isinf( px ) )
				continue;

			if ( isnan( py ) || isinf( py ) )
				continue;

			if ( isnan( pz ) || isinf( pz ) )
				continue;

			indices.push_back( idx );
			break;

		}

		for ( int y = cloud.height - 1; y >= 0; y-- ) {

			int idx = y * cloud.width + x;

			// if not nan, push back and break
			const pcl::PointXYZRGB& p = cloud.points[idx];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if ( isnan( px ) || isinf( px ) )
				continue;

			if ( isnan( py ) || isinf( py ) )
				continue;

			if ( isnan( pz ) || isinf( pz ) )
				continue;

			indices.push_back( idx );
			break;

		}

	}

}


template <typename T>
void MultiResolutionColorSurfelMap<T>::findVirtualBorderPoints( const pcl::PointCloud< pcl::PointXYZRGB >& cloud, std::vector< int >& indices ) {

	// detect background points at depth jumps
	// determine first image points from the borders that are not nan => use 0 depth beyond borders

	const float depthJumpRatio = 0.9f*0.9f;
	const float invDepthJumpRatio = 1.f/depthJumpRatio;

	indices.reserve( cloud.points.size() );

	// horizontally
	int idx = -1;
	for ( unsigned int y = 0; y < cloud.height; y++ ) {

		float lastDepth2 = 0.0;
		int lastIdx = -1;

		for ( unsigned int x = 0; x < cloud.width; x++ ) {

			idx++;

			// if not nan, push back and break
			const pcl::PointXYZRGB& p = cloud.points[idx];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if( isnan( px ) ) {
				continue;
			}

			// check for depth jumps
			float depth2 = px*px+py*py+pz*pz;
			float ratio = lastDepth2 / depth2;
			if( ratio < depthJumpRatio ) {
				indices.push_back( idx );
			}
			if( ratio > invDepthJumpRatio ) {
				indices.push_back( lastIdx );
			}

			lastIdx = idx;

			lastDepth2 = depth2;

		}

		if( lastIdx >= 0 )
			indices.push_back( lastIdx );

	}


	// vertically
	for ( unsigned int x = 0; x < cloud.width; x++ ) {

		float lastDepth2 = 0.0;
		int lastIdx = -1;

		for ( unsigned int y = 0; y < cloud.height; y++ ) {

			int idx = y * cloud.width + x;

			// if not nan, push back and break
			const pcl::PointXYZRGB& p = cloud.points[idx];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if( isnan( px ) )
				continue;

			// check for depth jumps
			float depth2 = px*px+py*py+pz*pz;
			float ratio = lastDepth2 / depth2;
			if( ratio < depthJumpRatio ) {
				indices.push_back( idx );
			}
			if( ratio > invDepthJumpRatio ) {
				indices.push_back( lastIdx );
			}

			lastIdx = idx;

			lastDepth2 = depth2;

		}

		if( lastIdx >= 0 )
			indices.push_back( lastIdx );

	}

}


template <typename T>
void MultiResolutionColorSurfelMap<T>::findForegroundBorderPoints( const pcl::PointCloud< pcl::PointXYZRGB >& cloud, std::vector< int >& indices ) {

	// detect foreground points at depth jumps
	// determine first image points from the borders that are not nan => use 0 depth beyond borders

	const float depthJumpRatio = 0.9f*0.9f;
	const float invDepthJumpRatio = 1.f/depthJumpRatio;

	indices.clear();
	indices.reserve( cloud.points.size() );

	// horizontally
	int idx = -1;
	for ( unsigned int y = 0; y < cloud.height; y++ ) {

		float lastDepth2 = 0.0;
		int lastIdx = -1;

		for ( unsigned int x = 0; x < cloud.width; x++ ) {

			idx++;

			// if not nan, push back and break
			const pcl::PointXYZRGB& p = cloud.points[idx];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if( isnan( px ) ) {
				continue;
			}

			// check for depth jumps
			float depth2 = px*px+py*py+pz*pz;
			float ratio = lastDepth2 / depth2;
			if( ratio < depthJumpRatio ) {
				if( lastIdx != -1 )
					indices.push_back( lastIdx );
			}
			if( ratio > invDepthJumpRatio ) {
				indices.push_back( idx );
			}

			lastIdx = idx;

			lastDepth2 = depth2;

		}

	}


	// vertically
	for ( unsigned int x = 0; x < cloud.width; x++ ) {

		float lastDepth2 = 0.0;
		int lastIdx = -1;

		for ( unsigned int y = 0; y < cloud.height; y++ ) {

			int idx = y * cloud.width + x;

			// if not nan, push back and break
			const pcl::PointXYZRGB& p = cloud.points[idx];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if( isnan( px ) )
				continue;

			// check for depth jumps
			float depth2 = px*px+py*py+pz*pz;
			float ratio = lastDepth2 / depth2;
			if( ratio < depthJumpRatio ) {
				if( lastIdx != -1 )
					indices.push_back( lastIdx );
			}
			if( ratio > invDepthJumpRatio ) {
				indices.push_back( idx );
			}

			lastIdx = idx;

			lastDepth2 = depth2;

		}

	}

}

template <typename T>
void MultiResolutionColorSurfelMap<T>::findContourPoints( const pcl::PointCloud< pcl::PointXYZRGB >& cloud, std::vector< int >& indices ) {

	// detect foreground points at depth jumps
	// determine first image points from the borders that are not nan => use 0 depth beyond borders

	const float depthJumpRatio = 0.95f*0.95f;
	const float invDepthJumpRatio = 1.f/depthJumpRatio;

	indices.clear();
	indices.reserve( cloud.points.size() );

	// horizontally
	int idx = -1;
	for ( unsigned int y = 0; y < cloud.height; y++ ) {

		float lastDepth2 = 0.0;
		int lastIdx = -1;

		for ( unsigned int x = 0; x < cloud.width; x++ ) {

			idx++;

			// if not nan, push back and break
			const pcl::PointXYZRGB& p = cloud.points[idx];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if( isnan( px ) ) {
				continue;
			}

			// check for depth jumps
			float depth2 = px*px+py*py+pz*pz;
			float ratio = lastDepth2 / depth2;
			if( ratio < depthJumpRatio ) {
				if( lastIdx != -1 )
					indices.push_back( lastIdx );
				indices.push_back( idx );
			}
			if( ratio > invDepthJumpRatio ) {
				indices.push_back( idx );
				if( lastIdx != -1 )
					indices.push_back( lastIdx );
			}

			lastIdx = idx;

			lastDepth2 = depth2;

		}

	}


	// vertically
	for ( unsigned int x = 0; x < cloud.width; x++ ) {

		float lastDepth2 = 0.0;
		int lastIdx = -1;

		for ( unsigned int y = 0; y < cloud.height; y++ ) {

			int idx = y * cloud.width + x;

			// if not nan, push back and break
			const pcl::PointXYZRGB& p = cloud.points[idx];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if( isnan( px ) )
				continue;

			// check for depth jumps
			float depth2 = px*px+py*py+pz*pz;
			float ratio = lastDepth2 / depth2;
			if( ratio < depthJumpRatio ) {
				if( lastIdx != -1 )
					indices.push_back( lastIdx );
				indices.push_back( idx );
			}
			if( ratio > invDepthJumpRatio ) {
				indices.push_back( idx );
				if( lastIdx != -1 )
					indices.push_back( lastIdx );
			}

			lastIdx = idx;

			lastDepth2 = depth2;

		}

	}

}


template <typename T>
void MultiResolutionColorSurfelMap<T>::clearAtPoints( const pcl::PointCloud< pcl::PointXYZRGB >& cloud, const std::vector< int >& indices ) {

	Eigen::Vector3d sensorOrigin;
	for ( int i = 0; i < 3; i++ )
		sensorOrigin( i ) = cloud.sensor_origin_( i );

	const double max_dist = MAX_VIEWDIR_DIST;

	// go through the point cloud and remove surfels
	for ( unsigned int i = 0; i < indices.size(); i++ ) {

		const pcl::PointXYZRGB& p = cloud.points[indices[i]];
		const float x = p.x;
		const float y = p.y;
		const float z = p.z;

		if ( isnan( x ) || isinf( x ) )
			continue;

		if ( isnan( y ) || isinf( y ) )
			continue;

		if ( isnan( z ) || isinf( z ) )
			continue;

		Eigen::Matrix< double, 3, 1 > pos;
		pos( 0 ) = p.x;
		pos( 1 ) = p.y;
		pos( 2 ) = p.z;

		Eigen::Vector3d viewDirection = pos - sensorOrigin;
		const double viewDistance = viewDirection.norm();

		if ( viewDistance < 1e-10 )
			continue;

		viewDirection = viewDirection / viewDistance;

		// traverse tree and clear all surfels that include this points
		spatialaggregate::OcTreeKey< float, T > position = octree_->getKey( p.getVector4fMap() );
		spatialaggregate::OcTreeNode< float, T >* n = octree_->root_;
		while ( n ) {


			for( unsigned int k = 0; k < n->value_.numberOfSurfels; k++ ) {
				const double dist = viewDirection.dot( n->value_.surfels_[k].initial_view_dir_ );
				if( dist > max_dist ) {
					n->value_.surfels_[k].clear();
				}
			}

			n = n->children_[n->getOctant( position )];
		}

	}

}

template <typename T>
void MultiResolutionColorSurfelMap<T>::markNoUpdateAtPoints( const pcl::PointCloud< pcl::PointXYZRGB >& cloud, const std::vector< int >& indices ) {

	Eigen::Vector3d sensorOrigin;
	for ( int i = 0; i < 3; i++ )
		sensorOrigin( i ) = cloud.sensor_origin_( i );

	const double max_dist = MAX_VIEWDIR_DIST;

	// go through the point cloud and remove surfels
	for ( unsigned int i = 0; i < indices.size(); i++ ) {

		const pcl::PointXYZRGB& p = cloud.points[indices[i]];
		const float x = p.x;
		const float y = p.y;
		const float z = p.z;

		if ( isnan( x ) || isinf( x ) )
			continue;

		if ( isnan( y ) || isinf( y ) )
			continue;

		if ( isnan( z ) || isinf( z ) )
			continue;

		Eigen::Matrix< double, 3, 1 > pos;
		pos( 0 ) = p.x;
		pos( 1 ) = p.y;
		pos( 2 ) = p.z;

		Eigen::Vector3d viewDirection = pos - sensorOrigin;
		const double viewDistance = viewDirection.norm();

		if ( viewDistance < 1e-10 )
			continue;

		viewDirection = viewDirection / viewDistance;

		// traverse tree and clear all surfels that include this points
		spatialaggregate::OcTreeKey< float, T > position = octree_->getKey( p.getVector4fMap() );
		spatialaggregate::OcTreeNode< float, T >* n = octree_->root_;
		while ( n ) {

			for( unsigned int k = 0; k < n->value_.numberOfSurfels; k++ ) {
				const double dist = viewDirection.dot( n->value_.surfels_[k].initial_view_dir_ );
				if( dist > max_dist ) {
					n->value_.surfels_[k].applyUpdate_ = false;
				}
			}

			n = n->children_[n->getOctant( position )];

		}

	}

}


template <typename T>
void MultiResolutionColorSurfelMap<T>::markBorderAtPoints( const pcl::PointCloud< pcl::PointXYZRGB >& cloud, const std::vector< int >& indices ) {

	Eigen::Vector3d sensorOrigin;
	for ( int i = 0; i < 3; i++ )
		sensorOrigin( i ) = cloud.sensor_origin_( i );

	const double max_dist = MAX_VIEWDIR_DIST;

	for ( unsigned int i = 0; i < indices.size(); i++ ) {

		const pcl::PointXYZRGB& p = cloud.points[indices[i]];
		const float x = p.x;
		const float y = p.y;
		const float z = p.z;

		if ( isnan( x ) || isinf( x ) )
			continue;

		if ( isnan( y ) || isinf( y ) )
			continue;

		if ( isnan( z ) || isinf( z ) )
			continue;

		Eigen::Matrix< double, 3, 1 > pos;
		pos( 0 ) = p.x;
		pos( 1 ) = p.y;
		pos( 2 ) = p.z;

		Eigen::Vector3d viewDirection = pos - sensorOrigin;
		const double viewDistance = viewDirection.norm();

		if ( viewDistance < 1e-10 )
			continue;

		viewDirection = viewDirection / viewDistance;

		spatialaggregate::OcTreeKey< float, T > position = octree_->getKey( p.getVector4fMap() );
		spatialaggregate::OcTreeNode< float, T >* n = octree_->root_;
		while ( n ) {

			n->value_.border_ = true;

			n = n->children_[n->getOctant( position )];

		}

	}

}


struct MarkBorderInfo {
	Eigen::Vector3d viewpoint;
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

template <typename T>
void markBorderFromViewpointFunction( spatialaggregate::OcTreeNode< float, T>* current, spatialaggregate::OcTreeNode< float, T >* next, void* data ) {

	MarkBorderInfo* info = (MarkBorderInfo*) data;

//	current->value_.border_ = false;

	for( unsigned int i = 0; i < 6; i++ ) {

		const ColorSurfel& surfel = current->value_.surfels_[i];

		if( surfel.num_points_ < MIN_SURFEL_POINTS )
			continue;

		Eigen::Vector3d viewDirection = info->viewpoint - surfel.mean_.block<3,1>(0,0);
		viewDirection.normalize();

		double cangle = viewDirection.dot( surfel.normal_ );

		if( cangle < 0.0 ) {
			current->value_.border_ = true;
//			for( unsigned int n = 0; n < 27; n++ )
//				if( current->neighbors_[n] )
//					current->neighbors_[n]->value_.border_ = true;
		}

	}

}

template <typename T>
void MultiResolutionColorSurfelMap<T>::markBorderFromViewpoint( const Eigen::Vector3d& viewpoint ) {

	MarkBorderInfo info;
	info.viewpoint = viewpoint;

	clearBorderFlag();
	octree_->root_->sweepDown( &info, &markBorderFromViewpointFunction );

}

template <typename T>
inline void MultiResolutionColorSurfelMap<T>::clearBorderFlagFunction( spatialaggregate::OcTreeNode< float, T >* current, spatialaggregate::OcTreeNode< float, T >* next, void* data ) {

	current->value_.border_ = false;

}

template <typename T>
void MultiResolutionColorSurfelMap<T>::clearBorderFlag() {

	octree_->root_->sweepDown( NULL, &clearBorderFlagFunction );

}

template <typename T>
void MultiResolutionColorSurfelMap<T>::clearUpdateSurfelsAtPoints( const pcl::PointCloud< pcl::PointXYZRGB >& cloud, const std::vector< int >& indices ) {

	Eigen::Vector3d sensorOrigin;
	for ( int i = 0; i < 3; i++ )
		sensorOrigin( i ) = cloud.sensor_origin_( i );

	const double max_dist = MAX_VIEWDIR_DIST;

	// go through the point cloud and remove surfels
	for ( unsigned int i = 0; i < indices.size(); i++ ) {

		const pcl::PointXYZRGB& p = cloud.points[indices[i]];
		const float x = p.x;
		const float y = p.y;
		const float z = p.z;

		if ( isnan( x ) )
			continue;

		Eigen::Matrix< double, 3, 1 > pos;
		pos( 0 ) = p.x;
		pos( 1 ) = p.y;
		pos( 2 ) = p.z;

		Eigen::Vector3d viewDirection = pos - sensorOrigin;
		const double viewDistance = viewDirection.norm();

		if ( viewDistance < 1e-10 )
			continue;

		viewDirection = viewDirection / viewDistance;

		// traverse tree and clear all surfels that include this points
		spatialaggregate::OcTreeKey< float, T > position = octree_->getKey( p.getVector4fMap() );
		spatialaggregate::OcTreeNode< float, T >* n = octree_->root_;
		while ( n ) {

			for( unsigned int k = 0; k < n->value_.numberOfSurfels; k++ ) {
				const double dist = viewDirection.dot( n->value_.surfels_[k].initial_view_dir_ );
				if( dist > max_dist ) {
					if( !n->value_.surfels_[k].up_to_date_ ) {
						n->value_.surfels_[k].clear();
					}
				}
			}

			n = n->children_[n->getOctant( position )];

		}

	}

}


template <typename T>
void MultiResolutionColorSurfelMap<T>::markUpdateAllSurfels() {

	octree_->root_->sweepDown( NULL, &markUpdateAllSurfelsFunction );

}

template <typename T>
inline void MultiResolutionColorSurfelMap<T>::markUpdateAllSurfelsFunction( spatialaggregate::OcTreeNode< float, T >* current, spatialaggregate::OcTreeNode< float, T >* next, void* data ) {

	for ( unsigned int i = 0; i < current->value_.numberOfSurfels; i++ )
		current->value_.surfels_[i].applyUpdate_ = true;

}


template <typename T>
void MultiResolutionColorSurfelMap<T>::markUpdateImprovedEffViewDistSurfels( const Eigen::Vector3f& viewPosition ) {

	Eigen::Vector3d viewPos = viewPosition.cast<double>();
	octree_->root_->sweepDown( &viewPos, &markUpdateImprovedEffViewDistSurfelsFunction );

}

template <typename T>
inline void MultiResolutionColorSurfelMap<T>::markUpdateImprovedEffViewDistSurfelsFunction( spatialaggregate::OcTreeNode<float, T>* current, spatialaggregate::OcTreeNode<float, T>* next, void* data ) {

	const Eigen::Vector3d& viewPos = *( (Eigen::Vector3d*) data );

	for ( unsigned int i = 0; i < current->value_.numberOfSurfels; i++ ) {

		MultiResolutionColorSurfelMap<T>::Surfel& surfel = current->value_.surfels_[i];

		// do we have to switch the flag?
		if( !surfel.applyUpdate_ ) {

			Eigen::Vector3d viewDir = surfel.mean_.block<3,1>(0,0) - viewPos;
			float effViewDist = viewDir.dot(surfel.initial_view_dir_) / viewDir.squaredNorm();
			if( effViewDist > surfel.eff_view_dist_ ) // > since it's inv eff view dist
				surfel.applyUpdate_ = true;

		}

	}

}


template <typename T>
void MultiResolutionColorSurfelMap<T>::evaluateSurfels() {

	octree_->root_->sweepDown( NULL, &evaluateNormalsFunction );
	octree_->root_->sweepDown( NULL, &evaluateSurfelsFunction );

}

template <typename T>
void MultiResolutionColorSurfelMap<T>::unevaluateSurfels() {

	octree_->root_->sweepDown( NULL, &unevaluateSurfelsFunction );

}

template <typename T>
void MultiResolutionColorSurfelMap<T>::setApplyUpdate( bool v ) {

	octree_->root_->sweepDown( &v, &setApplyUpdateFunction );

}

template <typename T>
void MultiResolutionColorSurfelMap<T>::setUpToDate( bool v ) {

	octree_->root_->sweepDown( &v, &setUpToDateFunction );

}

template <typename T>
void MultiResolutionColorSurfelMap<T>::clearUnstableSurfels() {

	octree_->root_->sweepDown( NULL, &clearUnstableSurfelsFunction );

}

template <typename T>
void MultiResolutionColorSurfelMap<T>::clearRobustChangeFlag() {

	octree_->root_->sweepDown( NULL, &clearRobustChangeFlagFunction );

}


template <typename T>
void MultiResolutionColorSurfelMap<T>::setRobustChangeFlag() {

	octree_->root_->sweepDown( NULL, &setRobustChangeFlagFunction );

}

template <typename T>
void MultiResolutionColorSurfelMap<T>::evaluateSurfelPairRelations() {

	octree_->root_->sweepDown( (void*)this, &evaluateSurfelPairRelationsFunction );

}

template <typename T>
void MultiResolutionColorSurfelMap<T>::clearAssociatedFlag() {

	octree_->root_->sweepDown( NULL, &clearAssociatedFlagFunction );

}

template <typename T>
void MultiResolutionColorSurfelMap<T>::distributeAssociatedFlag() {

	octree_->root_->sweepDown( NULL, &distributeAssociatedFlagFunction );

}

template <typename T>
void MultiResolutionColorSurfelMap<T>::clearAssociations() {

	octree_->root_->sweepDown( NULL, &clearAssociationsFunction );

}

template <typename T>
inline void MultiResolutionColorSurfelMap<T>::clearAssociationsFunction( spatialaggregate::OcTreeNode< float, T >* current, spatialaggregate::OcTreeNode< float, T >* next, void* data ) {
	if( current->value_.associated_ != -1 )
		current->value_.associated_ = 1;
	current->value_.association_ = NULL;
}

template <typename T>
bool MultiResolutionColorSurfelMap<T>::pointInForeground( const Eigen::Vector3f& position, const cv::Mat& image_depth, const cv::Point2f imagePoint, float scale, float jumpThreshold ) {

	float queryDepth = position.norm();

	int scale05 = ceil( 0.5f * scale );

	cv::Rect r;
	r.x = (int) floor( imagePoint.x - scale05 );
	r.y = (int) floor( imagePoint.y - scale05 );
	r.width = 2 * scale05;
	r.height = 2 * scale05;

	if ( r.x < 0 ) {
		r.width += r.x;
		r.x = 0;
	}

	if ( r.y < 0 ) {
		r.height += r.y;
		r.y = 0;
	}

	if ( r.x + r.width > image_depth.cols )
		r.width = image_depth.cols - r.x;

	if ( r.y + r.height > image_depth.rows )
		r.height = image_depth.rows - r.y;

	cv::Mat patch = image_depth( r );

	// find correponding point for query point in image
	float bestDist = 1e10f;
	int bestX = -1, bestY = -1;
	for ( int y = 0; y < patch.rows; y++ ) {
		for ( int x = 0; x < patch.cols; x++ ) {
			const float depth = patch.at< float > ( y, x );
			if ( !isnan( depth ) ) {
				float dist = fabsf( queryDepth - depth );
				if ( dist < bestDist ) {
					bestDist = dist;
					bestX = x;
					bestY = y;
				}
			}

		}
	}

	// find depth jumps to the foreground in horizontal, vertical, and diagonal directions
	//	cv::Mat img_show = image_depth.clone();

	for ( int dy = -1; dy <= 1; dy++ ) {
		for ( int dx = -1; dx <= 1; dx++ ) {

			if ( dx == 0 && dy == 0 )
				continue;

			float trackedDepth = queryDepth;
			for ( int y = bestY + dy, x = bestX + dx; y >= 0 && y < patch.rows && x >= 0 && x < patch.cols; y += dy, x += dx ) {

				const float depth = patch.at< float > ( y, x );
				//				img_show.at<float>(r.y+y,r.x+x) = 0.f;
				if ( !isnan( depth ) ) {

					if ( trackedDepth - depth > jumpThreshold ) {
						return false;
					}

					trackedDepth = depth;

				}

			}

		}
	}

	return true;
}

template <typename T>
void MultiResolutionColorSurfelMap<T>::buildShapeTextureFeatures() {

	octree_->root_->sweepDown( NULL, &buildSimpleShapeTextureFeatureFunction );
	octree_->root_->sweepDown( NULL, &buildAgglomeratedShapeTextureFeatureFunction );

}

template <typename T>
inline void MultiResolutionColorSurfelMap<T>::buildSimpleShapeTextureFeatureFunction( spatialaggregate::OcTreeNode<float, T>* current, spatialaggregate::OcTreeNode<float, T>* next, void* data ) {

	for( unsigned int i = 0; i < current->value_.numberOfSurfels; i++ ) {

		current->value_.surfels_[i].simple_shape_texture_features_.initialize();

		if( current->value_.surfels_[i].num_points_ < MIN_SURFEL_POINTS )
			continue;

		current->value_.surfels_[i].simple_shape_texture_features_.add( &current->value_.surfels_[i], &current->value_.surfels_[i], current->value_.surfels_[i].num_points_ );

		for( unsigned int n = 0; n < 27; n++ ) {

			if( n == 13 ) // pointer to this node
				continue;

			if( current->neighbors_[n] ) {

				if( current->neighbors_[n]->value_.surfels_[i].num_points_ < MIN_SURFEL_POINTS )
					continue;

				current->value_.surfels_[i].simple_shape_texture_features_.add( &current->value_.surfels_[i], &current->neighbors_[n]->value_.surfels_[i], current->neighbors_[n]->value_.surfels_[i].num_points_ );

			}

		}

	}

}

template <typename T>
inline void MultiResolutionColorSurfelMap<T>::buildAgglomeratedShapeTextureFeatureFunction( spatialaggregate::OcTreeNode<float, T>* current, spatialaggregate::OcTreeNode<float, T>* next, void* data ) {

	const float neighborFactor = 0.1f;

	for( unsigned int i = 0; i < current->value_.numberOfSurfels; i++ ) {

		current->value_.surfels_[i].agglomerated_shape_texture_features_ = current->value_.surfels_[i].simple_shape_texture_features_;

		if( current->value_.surfels_[i].num_points_ < MIN_SURFEL_POINTS )
			continue;

		for( unsigned int n = 0; n < 27; n++ ) {

			if( n == 13 ) // pointer to this node
				continue;

			if( current->neighbors_[n] ) {

				if( current->neighbors_[n]->value_.surfels_[i].num_points_ < MIN_SURFEL_POINTS ) {
					continue;
				}

				current->value_.surfels_[i].agglomerated_shape_texture_features_.add( current->neighbors_[n]->value_.surfels_[i].simple_shape_texture_features_, current->neighbors_[n]->value_.surfels_[i].simple_shape_texture_features_.num_points_ * neighborFactor );

			}

		}

		if( current->value_.surfels_[i].agglomerated_shape_texture_features_.num_points_ > 0.5f ) {
			float inv_num = 1.f / current->value_.surfels_[i].agglomerated_shape_texture_features_.num_points_;
			current->value_.surfels_[i].agglomerated_shape_texture_features_.shape_ *= inv_num;
			current->value_.surfels_[i].agglomerated_shape_texture_features_.texture_ *= inv_num;
		}

		current->value_.surfels_[i].agglomerated_shape_texture_features_.num_points_ = 1.f;

	}

}

template <typename T>
void MultiResolutionColorSurfelMap<T>::clearAssociationDist() {
	octree_->root_->sweepDown( NULL, &clearAssociationDistFunction );
}


template <typename T>
inline void MultiResolutionColorSurfelMap<T>::clearAssociationDistFunction(spatialaggregate::OcTreeNode<float, T>* current, spatialaggregate::OcTreeNode<float, T>* next, void* data) {
	for ( unsigned int i = 0; i < current->value_.numberOfSurfels; i++ ) {
		current->value_.surfels_[i].assocDist_ = std::numeric_limits<float>::max();
	}
}


template <typename T>
inline void MultiResolutionColorSurfelMap<T>::setApplyUpdateFunction( spatialaggregate::OcTreeNode< float, T >* current, spatialaggregate::OcTreeNode< float, T >* next, void* data ) {
	bool v = *((bool*) data);
	for ( unsigned int i = 0; i < current->value_.numberOfSurfels; i++ ) {
		if( current->value_.surfels_[i].num_points_ >= MIN_SURFEL_POINTS ) {
			current->value_.surfels_[i].applyUpdate_ = v;
		}
	}
}

template <typename T>
inline void MultiResolutionColorSurfelMap<T>::setUpToDateFunction( spatialaggregate::OcTreeNode< float, T >* current, spatialaggregate::OcTreeNode< float, T >* next, void* data ) {
	bool v = *((bool*) data);
	for ( unsigned int i = 0; i < current->value_.numberOfSurfels; i++ ) {
		current->value_.surfels_[i].up_to_date_ = v;
	}
}

template <typename T>
inline void MultiResolutionColorSurfelMap<T>::clearUnstableSurfelsFunction( spatialaggregate::OcTreeNode< float, T >* current, spatialaggregate::OcTreeNode< float, T >* next, void* data ) {
	for ( unsigned int i = 0; i < current->value_.numberOfSurfels; i++ ) {
		if( current->value_.surfels_[i].num_points_ < MIN_SURFEL_POINTS ) {
			// reinitialize
			current->value_.surfels_[i].up_to_date_ = false;
			current->value_.surfels_[i].mean_.setZero();
			current->value_.surfels_[i].cov_.setZero();
			current->value_.surfels_[i].num_points_ = 0;
			current->value_.surfels_[i].became_robust_ = false;
			current->value_.surfels_[i].applyUpdate_ = true;
		}
	}
}


template <typename T>
inline void MultiResolutionColorSurfelMap<T>::clearRobustChangeFlagFunction( spatialaggregate::OcTreeNode< float, T >* current, spatialaggregate::OcTreeNode< float, T >* next, void* data ) {
	for ( unsigned int i = 0; i < current->value_.numberOfSurfels; i++ ) {
		current->value_.surfels_[i].became_robust_ = false;
	}
}

template <typename T>
inline void MultiResolutionColorSurfelMap<T>::setRobustChangeFlagFunction( spatialaggregate::OcTreeNode< float, T >* current, spatialaggregate::OcTreeNode< float, T >* next, void* data ) {
	for ( unsigned int i = 0; i < current->value_.numberOfSurfels; i++ ) {
		current->value_.surfels_[i].became_robust_ = true;
	}
}


template <typename T>
inline void MultiResolutionColorSurfelMap<T>::evaluateSurfelPairRelationsFunction( spatialaggregate::OcTreeNode< float, T >* current, spatialaggregate::OcTreeNode< float, T >* next, void* data ) {
	MultiResolutionColorSurfelMap* map = (MultiResolutionColorSurfelMap*) data;
	for ( unsigned int i = 0; i < current->value_.numberOfSurfels; i++ ) {
		if( current->value_.surfels_[i].became_robust_ && current->value_.surfels_[i].num_points_ > NUM_SURFEL_POINTS_ROBUST && current->depth_ > PAIR_RELATION_NEIGHBORHOOD_UPLAYERS ) {

			// find all nodes on query depth within query volume
			const double searchRadius = 0.5 * (current->resolution()) * pow( 2.0, ((double)PAIR_RELATION_NEIGHBORHOOD_UPLAYERS) );
			//( std::vector< OcTreeNode< CoordType, ValueType >* >& points, const OcTreePosition< CoordType >& minPosition, const OcTreePosition< CoordType >& maxPosition, int searchDepth, bool higherDepthLeaves, int maxDepth ) {
			std::list< spatialaggregate::OcTreeNode< float, T >* > nodes;
			Eigen::Matrix< float, 4, 1 > minPosition, maxPosition;
			minPosition[0] = current->value_.surfels_[i].mean_(0) - searchRadius;
			minPosition[1] = current->value_.surfels_[i].mean_(1) - searchRadius;
			minPosition[2] = current->value_.surfels_[i].mean_(2) - searchRadius;
			maxPosition[0] = current->value_.surfels_[i].mean_(0) + searchRadius;
			maxPosition[1] = current->value_.surfels_[i].mean_(1) + searchRadius;
			maxPosition[2] = current->value_.surfels_[i].mean_(2) + searchRadius;
			map->octree_->getAllNodesInVolumeOnDepth( nodes, minPosition, maxPosition, current->depth_, false );

			// precalculate signature information from node list:
			// - for each neighbor surfel, project surfel onto normal plane, store statistics/relative position in list
			// - when traversing the pair relations, transform statistics to local coordinate frame
			//   and extract signature
			ContextSignatureInformation csi;
			map->prepareContextSignatureInformation( csi, current, i, nodes, current->depth_ );

			// build up shape/texture context concurrently
			// TODO: no context / surfel pair when nodes at image borders involved?
			map->traverseAndBuildSurfelPairRelationsOnDepth( current, i, csi, nodes, current->depth_ );

			current->value_.surfels_[i].became_robust_ = false;

			// assign surfel a new idx if its still unassigned
			if( current->value_.surfels_[i].idx_ == -1 )
				current->value_.surfels_[i].idx_ = map->last_pair_surfel_idx_++;

		}
	}
}

template <typename T>
inline int MultiResolutionColorSurfelMap<T>::traverseAndBuildSurfelPairRelationsOnDepth( spatialaggregate::OcTreeNode< float, T >* src, unsigned int surfelIdx, const ContextSignatureInformation& csi, std::list< spatialaggregate::OcTreeNode< float, T >* > nodes, unsigned int depth ) {

	int numFoundPairs = 0;

	const double nodeSize = src->resolution();
	const double max_distance = 0.5*nodeSize * pow( 2.0, ((double)PAIR_RELATION_NEIGHBORHOOD_UPLAYERS) );
	const double min_distance = 2.0*nodeSize; // to guarantee some diversity and robustness in the difference vectors
	const double min_distance2 = min_distance*min_distance;
	const double max_distance2 = max_distance*max_distance;

	for( typename std::list< spatialaggregate::OcTreeNode< float, T >* >::iterator it = nodes.begin(); it != nodes.end(); ++it ) {

		spatialaggregate::OcTreeNode< float, T >* dst = *it;

		// TODO: with the new one-way signature, do not check became-robust-flag of dst surfel, such that each surfel may generate pairs with itself as source

		// build pair relations if dst != current and dst surfel did not become robust lately
		for ( unsigned int i = 0; i < dst->value_.numberOfSurfels; i++ ) {
			if( /*!dst->value_.surfels_[i].became_robust_ &&*/ dst->value_.surfels_[i].num_points_ > NUM_SURFEL_POINTS_ROBUST ) {

				double dist2 = (src->value_.surfels_[surfelIdx].mean_.template block<3,1>(0,0) - dst->value_.surfels_[i].mean_.template block<3,1>(0,0)).squaredNorm();

				if( dist2 < min_distance2 || dist2 > max_distance2 )
					continue;

				// TODO: "upsample" the octree sampling rate to become robust for discretization effects
				// idea: instead of upsampling, we could estimate normals in larger surroundings,
				// then the estimate of the normal is also not that strongly affected by the localization of the mean
				// this is simpler to implement, but requires an additional finer resolution in the octree..
				// problem: we may not have enough points to pass the test above on this finer resolution

				// TODO: context signature: dont use as key because of occlusions! use later to sort out bad matches between surfels with the hamming distance

				// retrieve signature and store it in hash map for current depth
				MultiResolutionColorSurfelMap::SurfelPairSignature signature = buildSurfelPairRelation( src->value_.surfels_[surfelIdx], dst->value_.surfels_[i], csi, max_distance, nodeSize );
				SurfelPair pair( &src->value_.surfels_[surfelIdx], &dst->value_.surfels_[i], signature.reference_pose_, signature.context_signature_ );
				surfel_pair_map_[depth][signature.getKey()].push_back( pair );

//				// retrieve opposite direction signature and store it in hash map for current depth
//				SurfelPairSignature signature2 = buildSurfelPairRelation( dst->value_.surfels_[i], src->value_.surfels_[surfelIdx], max_distance );
//				SurfelPair pair2( &dst->value_.surfels_[i], &src->value_.surfels_[surfelIdx], signature2.reference_pose_ );
//				surfel_pair_map_[depth][signature2.getKey()].push_back( pair2 );

//				ROS_ERROR("%lf %lf %i %i", (double)signature.getKey(), (double)signature2.getKey(), surfel_pair_map_[depth][signature.getKey()].size(), surfel_pair_map_[depth][signature2.getKey()].size() );

//				if( signature.getKey() == signature2.getKey() ) {
//					std::cout << "same:\n";
//					std::cout << signature.shape_signature_ << "\n" << signature.color_signature_ << "\n";
//					std::cout << signature2.shape_signature_ << "\n" << signature2.color_signature_ << "\n";
//				}

				// assign surfel a new idx if it's still unassigned
				if( dst->value_.surfels_[i].idx_ == -1 )
					dst->value_.surfels_[i].idx_ = last_pair_surfel_idx_++;

				numFoundPairs++;

			}
		}

	}

	return numFoundPairs;

}

#define DEBUG_CONTEXT_SIGNATURE 0

template <typename T>
inline void MultiResolutionColorSurfelMap<T>::prepareContextSignatureInformation( ContextSignatureInformation& csi, spatialaggregate::OcTreeNode< float, T >* src, unsigned int surfelIdx, std::list< spatialaggregate::OcTreeNode< float, T >* > dstNodes, unsigned int depth ) {

	const double resolution = src->resolution();
	const double max_distance = 0.5*resolution * pow( 2.0, ((double)PAIR_RELATION_NEIGHBORHOOD_UPLAYERS) );
	const double max_distance2 = max_distance*max_distance;

	const double threshShape = 1.0 * resolution;
	const double threshL = 0.1;
	const double threshAlpha = 0.1;
	const double threshBeta = 0.1;

	Eigen::Vector4d centerRadii;
	centerRadii(0) = threshShape;
	centerRadii(1) = threshL;
	centerRadii(2) = threshAlpha;
	centerRadii(3) = threshBeta;

	csi = ContextSignatureInformation( max_distance, centerRadii, 0.25*centerRadii );

	// determine local reference axes for the src surfel's normal
	Eigen::Vector3d refAxis1, refAxis2;
	const double anx = fabsf(src->value_.surfels_[surfelIdx].normal_(0));
	const double any = fabsf(src->value_.surfels_[surfelIdx].normal_(1));
	const double anz = fabsf(src->value_.surfels_[surfelIdx].normal_(2));
	if( anx <= any && anx <= anz )
		refAxis1 = Eigen::Vector3d( 1, 0, 0 );
	else if( any <= anx && any <= anz )
		refAxis1 = Eigen::Vector3d( 0, 1, 0 );
	else
		refAxis1 = Eigen::Vector3d( 0, 0, 1 );

	refAxis1 = src->value_.surfels_[surfelIdx].normal_.cross( refAxis1 );
	refAxis1.normalize();
	refAxis2 = src->value_.surfels_[surfelIdx].normal_.cross( refAxis1 );
	refAxis2.normalize();

	csi.refAxis1 = refAxis1;
	csi.refAxis2 = refAxis2;

#if DEBUG_CONTEXT_SIGNATURE
	static int idx = 0;
	char str[255];
	sprintf(str, "descriptor/positions%i.dat", idx++);
	std::ofstream outfile( str );

	outfile << src->value_.surfels_[surfelIdx].normal_(0) << " " << src->value_.surfels_[surfelIdx].normal_(1) << " " << src->value_.surfels_[surfelIdx].normal_(2) << " 0 0 0 0 0 0\n";
	outfile << refAxis1(0) << " " << refAxis1(1) << " " << refAxis1(2) << " 0 0 0 0 0 0\n";
	outfile << refAxis2(0) << " " << refAxis2(1) << " " << refAxis2(2) << " 0 0 0 0 0 0\n";

	MultiResolutionColorSurfelMap<T>::Surfel* dst = NULL;
#endif

	// extract relative signature information of neighbor surfels
	for( typename std::list< spatialaggregate::OcTreeNode< float, T >* >::iterator it = dstNodes.begin(); it != dstNodes.end(); ++it ) {

		if( (*it) == src )
			continue;

		if( (*it)->depth_ != depth )
			continue;

		// view direction: only use surfels that match in view direction.. just the same index (surfelIdx)
		if( (*it)->value_.surfels_[surfelIdx].num_points_ < MIN_SURFEL_POINTS )
			continue;

		double dist2 = (src->value_.surfels_[surfelIdx].mean_.template block<3,1>(0,0) - (*it)->value_.surfels_[surfelIdx].mean_.template block<3,1>(0,0)).squaredNorm();

		if( dist2 > max_distance2 )
			continue;

#if DEBUG_CONTEXT_SIGNATURE
		if( !dst )
			dst = &(*it)->value_.surfels_[surfelIdx];
#endif

		// project surfel to the plane orthogonal to the src surfel's normal
		// plane is given by the local reference axes
		Eigen::Vector3d posDiff = (*it)->value_.surfels_[surfelIdx].mean_.template block<3,1>(0,0) - src->value_.surfels_[surfelIdx].mean_.template block<3,1>(0,0);
		Eigen::Vector2d localPos;
		localPos(0) = refAxis1.dot( posDiff );
		localPos(1) = refAxis2.dot( posDiff );

		// TODO: verhalten des deskriptors an unflachen stellen pruefen, wo sich die normale ungenau schaetzen laesst
		// calculate descriptor values for the surfel pair
		Eigen::Matrix< double, 4, 1 > descriptor;

		// shape: distance to normal plane at surfel
		double height = src->value_.surfels_[surfelIdx].normal_.dot( posDiff );

		descriptor(0) = height;

		// texture: Lab color distance between surfels
		Eigen::Vector3d colorDiff = (*it)->value_.surfels_[surfelIdx].mean_.template block<3,1>(3,0) - src->value_.surfels_[surfelIdx].mean_.template block<3,1>(3,0);
		descriptor.block<3,1>(1,0) = colorDiff;

//		csi.local_position_.push_back( localPos );
//		csi.descriptors_.push_back( descriptor );

		csi.grid_.add( localPos, descriptor );

#if DEBUG_CONTEXT_SIGNATURE
		outfile << posDiff(0) << " " << posDiff(1) << " " << posDiff(2) << " " << localPos(0) << " " << localPos(1) << " " << descriptor(0) << " " << descriptor(1) << " " << descriptor(2) << " " << descriptor(3) << "\n";
#endif

	}

#if DEBUG_CONTEXT_SIGNATURE
	if( 1 && dst ) {
		const double max_distance = 0.5 * (src->maxPosition.p[0] - src->minPosition.p[0]) * pow( 2.0, ((double)PAIR_RELATION_NEIGHBORHOOD_UPLAYERS) );
		buildContextSignature( src->value_.surfels_[surfelIdx], *dst, csi, max_distance, resolution, idx-1 );
	}
#endif



#if DEBUG_CONTEXT_SIGNATURE
	char str[255];
	sprintf(str, "descriptor/polargrid%i.dat", fileIdx);
	std::ofstream outfile( str );

	outfile << csi.grid_ << "\n";

	sprintf(str, "descriptor/polargridbin%i.dat", fileIdx);
	std::ofstream outfile2( str );
#endif




}

template <typename T>
inline typename MultiResolutionColorSurfelMap<T>::ContextSignature MultiResolutionColorSurfelMap<T>::buildContextSignature( const MultiResolutionColorSurfelMap<T>::Surfel& src, const MultiResolutionColorSurfelMap<T>::Surfel& dst, const ContextSignatureInformation& csi, double max_distance, double resolution, int fileIdx ) {

	// create binary descriptor by sampling and linear interpolation in the polar histogram
	MultiResolutionColorSurfelMap<T>::ContextSignature contextSignature;

#if DEBUG_CONTEXT_SIGNATURE
	outfile << grid << "\n";

	sprintf(str, "descriptor/polargridbin%i.dat", fileIdx);
	std::ofstream outfile2( str );
#endif

	// align local samples relative to the line between the surfels
	// (zero angle is along line from src to dst)

	Eigen::Vector3d refPosDiff = dst.mean_.block<3,1>(0,0) - src.mean_.block<3,1>(0,0);
	Eigen::Vector2d localRefPos;
	localRefPos(0) = csi.refAxis1.dot( refPosDiff );
	localRefPos(1) = csi.refAxis2.dot( refPosDiff );

#if DEBUG_CONTEXT_SIGNATURE
	outfile << localRefPos(0) << " " << localRefPos(1) << " " << max_distance << " 0 0 0 0 0 0 0 0 0 0 0 0 0\n";
#endif

	// ref angle towards x-dir
	double refAngle = atan2( localRefPos(1), localRefPos(0) );

	// binarize grid
	const double threshold = 0.5;


	csi.grid_.binarizeRotated( contextSignature.context_signature_, refAngle, threshold );

#if DEBUG_CONTEXT_SIGNATURE
	for( int r = 0; r < csi.grid_.radiusBins_; r++ ) {
		for( int a = 0; a < csi.grid_.angleBins_; a++ ) {

			int signature_idx = r*csi.grid_.angleBins_ + a;
			outfile2 << ((contextSignature.context_signature_[signature_idx*SIGNATURE_ELEMENTS+0] & (1)) ? 1 : 0) << " "
					<< ((contextSignature.context_signature_[signature_idx*SIGNATURE_ELEMENTS+1] & (1 << 1)) ? 1 : 0) << " "
					<< ((contextSignature.context_signature_[signature_idx*SIGNATURE_ELEMENTS+2] & (1 << 2)) ? 1 : 0) << " "
					<< ((contextSignature.context_signature_[signature_idx*SIGNATURE_ELEMENTS+3] & (1 << 3)) ? 1 : 0) << " "
					<< ((contextSignature.context_signature_[signature_idx*SIGNATURE_ELEMENTS+4] & (1 << 4)) ? 1 : 0) << " "
					<< ((contextSignature.context_signature_[signature_idx*SIGNATURE_ELEMENTS+5] & (1 << 5)) ? 1 : 0) << "\n";

		}
	}
#endif

	return contextSignature;


}

template <typename T>
inline typename MultiResolutionColorSurfelMap<T>::SurfelPairSignature MultiResolutionColorSurfelMap<T>::buildSurfelPairRelation( const MultiResolutionColorSurfelMap<T>::Surfel& src, const MultiResolutionColorSurfelMap<T>::Surfel& dst, const ContextSignatureInformation& csi, double max_distance, double resolution ) {

	// extract signature
	typename MultiResolutionColorSurfelMap<T>::SurfelPairSignature signature;

	// surflet pair relation as in "model globally match locally"
	Eigen::Vector3d p1 = src.mean_.block<3,1>(0,0);
	Eigen::Vector3d p2 = dst.mean_.block<3,1>(0,0);
	Eigen::Vector3d n1 = src.normal_;
	Eigen::Vector3d n2 = dst.normal_;

	Eigen::Vector3d d = p2-p1;
	Eigen::Vector3d d_normalized = d / d.norm();

	// normalize ranges to [0,1]
	signature.shape_signature_(0) = d.norm() / max_distance;
	signature.shape_signature_(1) = 0.5 * (n1.dot( d_normalized )+1.0);
	signature.shape_signature_(2) = 0.5 * (n2.dot( d_normalized )+1.0);
	signature.shape_signature_(3) = 0.5 * (n1.dot( n2 )+1.0);

	// color comparison with mean L alpha beta
	// normalize ranges to [0,1]
	signature.color_signature_ = dst.mean_.block<3,1>(3,0) - src.mean_.block<3,1>(3,0);
	signature.color_signature_(0) = 0.5 * (signature.color_signature_(0)+1.0); // L in [0,1]
	signature.color_signature_(1) = 0.25 * (signature.color_signature_(1)+2.0); // alpha in [-1,1]
	signature.color_signature_(2) = 0.25 * (signature.color_signature_(2)+2.0); // beta in [-1,1]

	// extract reference pose
	Eigen::Matrix3d referenceRot;
	referenceRot.setIdentity();
	referenceRot.block<3,1>(0,0) = n1;
	referenceRot.block<3,1>(0,1) = (n1.cross( d_normalized )).normalized();
	referenceRot.block<3,1>(0,2) = n1.cross( referenceRot.block<3,1>(0,1) );

	// reference pose represents transform from reference frame to map frame
	signature.reference_pose_.template block<3,1>(0,0) = p1;
	Eigen::Quaterniond q( referenceRot );
	signature.reference_pose_(3,0) = q.x();
	signature.reference_pose_(4,0) = q.y();
	signature.reference_pose_(5,0) = q.z();
	signature.reference_pose_(6,0) = q.w();

	signature.context_signature_ = buildContextSignature( src, dst, csi, max_distance, resolution );

	return signature;

}




template <typename T>
inline void MultiResolutionColorSurfelMap<T>::evaluateNormalsFunction( spatialaggregate::OcTreeNode< float, T >* current, spatialaggregate::OcTreeNode< float, T >* next, void* data ) {
	current->value_.evaluateNormals( current );
}

template <typename T>
inline void MultiResolutionColorSurfelMap<T>::evaluateSurfelsFunction( spatialaggregate::OcTreeNode< float, T >* current, spatialaggregate::OcTreeNode< float, T >* next, void* data ) {
	current->value_.evaluateSurfels();
}

template <typename T>
inline void MultiResolutionColorSurfelMap<T>::unevaluateSurfelsFunction( spatialaggregate::OcTreeNode< float, T >* current, spatialaggregate::OcTreeNode< float, T >* next, void* data ) {
	current->value_.unevaluateSurfels();
}

template <typename T>
inline void MultiResolutionColorSurfelMap<T>::clearAssociatedFlagFunction( spatialaggregate::OcTreeNode< float, T >* current, spatialaggregate::OcTreeNode< float, T >* next, void* data ) {
	if( current->value_.associated_ != -1 )
		current->value_.associated_ = 1;
}

template <typename T>
inline void MultiResolutionColorSurfelMap<T>::distributeAssociatedFlagFunction( spatialaggregate::OcTreeNode< float, T >* current, spatialaggregate::OcTreeNode< float, T >* next, void* data ) {

	for( unsigned int n = 0; n < 27; n++ ) {

		if( current->neighbors_[n] && current->neighbors_[n]->value_.associated_ == 0 ) {
			current->neighbors_[n]->value_.associated_ = 2;
		}

	}

}


template <typename T>
std::vector< unsigned int > MultiResolutionColorSurfelMap<T>::findInliers( const std::vector< unsigned int >& indices, const pcl::PointCloud<pcl::PointXYZRGB>& cloud, int maxDepth ) {

	std::vector< unsigned int > inliers;
	inliers.reserve( indices.size() );

	const float max_mahal_dist = 12.59f;

	const double inv_255 = 1.0 / 255.0;
	const float sqrt305 = 0.5f*sqrtf(3.f);

	Eigen::Vector3d sensorOrigin;
	for ( int i = 0; i < 3; i++ )
		sensorOrigin( i ) = cloud.sensor_origin_( i );

	// project each point into map and find inliers
	// go through the point cloud and add point information to map
	for ( unsigned int i = 0; i < indices.size(); i++ ) {

		const pcl::PointXYZRGB& p = cloud.points[indices[i]];
		const float x = p.x;
		const float y = p.y;
		const float z = p.z;

		if ( isnan( x ) )
			continue;

		float rgbf = p.rgb;
		unsigned int rgb = * ( reinterpret_cast< unsigned int* > ( &rgbf ) );
		unsigned int r = ( ( rgb & 0x00FF0000 ) >> 16 );
		unsigned int g = ( ( rgb & 0x0000FF00 ) >> 8 );
		unsigned int b = ( rgb & 0x000000FF );

		// HSL by Luminance and Cartesian Hue-Saturation (L-alpha-beta)
		float rf = inv_255*r, gf = inv_255*g, bf = inv_255*b;

		// RGB to L-alpha-beta:
		// normalize RGB to [0,1]
		// M := max( R, G, B )
		// m := min( R, G, B )
		// L := 0.5 ( M + m )
		// alpha := 0.5 ( 2R - G - B )
		// beta := 0.5 sqrt(3) ( G - B )
		float L = 0.5f * ( std::max( std::max( rf, gf ), bf ) + std::min( std::min( rf, gf ), bf ) );
		float alpha = 0.5f * ( 2.f*rf - gf - bf );
		float beta = sqrt305 * (gf-bf);


		Eigen::Matrix< double, 6, 1 > pos;
		pos( 0 ) = p.x;
		pos( 1 ) = p.y;
		pos( 2 ) = p.z;
		pos( 3 ) = L;
		pos( 4 ) = alpha;
		pos( 5 ) = beta;


		Eigen::Vector3d viewDirection = pos.block< 3, 1 > ( 0, 0 ) - sensorOrigin;
		viewDirection.normalize();

		Eigen::Vector4f pos4f = pos.block<4,1>(0,0).cast<float>();

		// lookup node for point
		spatialaggregate::OcTreeNode< float, T>* n = octree_->root_->findRepresentative( pos4f, maxDepth );

		MultiResolutionColorSurfelMap::Surfel* surfel = n->value_.getSurfel( viewDirection );
		if( surfel->num_points_ > MIN_SURFEL_POINTS ) {

			// inlier? check mahalanobis distance
			Eigen::Matrix< double, 6, 6 > invcov = surfel->cov_.inverse();
			Eigen::Matrix< double, 6, 1 > diff = surfel->mean_.block<6,1>(0,0) - pos;

			if( diff.dot( invcov * diff ) < max_mahal_dist ) {
				inliers.push_back( i );
			}

		}


	}

}




struct Visualize3DColorDistributionInfo {
	pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr;
	int viewDir, depth;
	bool random;
};


template <typename T>
void MultiResolutionColorSurfelMap<T>::visualize3DColorDistribution( pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr, int depth, int viewDir, bool random ) {

	Visualize3DColorDistributionInfo info;
	info.cloudPtr = cloudPtr;
	info.viewDir = viewDir;
	info.depth = depth;
	info.random = random;

	octree_->root_->sweepDown( &info, &visualize3DColorDistributionFunction );

}

template <typename T>
inline void MultiResolutionColorSurfelMap<T>::visualize3DColorDistributionFunction( spatialaggregate::OcTreeNode< float, T >* current, spatialaggregate::OcTreeNode< float, T >* next, void* data ) {

	Visualize3DColorDistributionInfo* info = (Visualize3DColorDistributionInfo*) data;

	if( (info->depth == -1 && current->type_ == spatialaggregate::OCTREE_BRANCHING_NODE) )
		return;

	if( info->depth >= 0 && current->depth_ != info->depth )
		return;

	if( current->depth_ < 10 )
		return;

//	std::cout << current->resolution() << "\n";

	Eigen::Matrix< float, 4, 1 > minPos = current->getMinPosition();
	Eigen::Matrix< float, 4, 1 > maxPos = current->getMaxPosition();


	// generate markers for histogram surfels
	for ( unsigned int i = 0; i < current->value_.numberOfSurfels; i++ ) {

		if( info->viewDir >= 0 && info->viewDir != i )
			continue;

		const MultiResolutionColorSurfelMap<T>::Surfel& surfel = current->value_.surfels_[i];

		if ( surfel.num_points_ < MIN_SURFEL_POINTS )
			continue;

		if( info->random ) {

			// samples N points from the normal distribution in mean and cov...
			unsigned int N = 100;

			// cholesky decomposition
			Eigen::Matrix< double, 6, 6 > cov = surfel.cov_;
			Eigen::LLT< Eigen::Matrix< double, 6, 6 > > chol = cov.llt();


			for ( unsigned int j = 0; j < N; j++ ) {

				Eigen::Matrix< double, 6, 1 > vec;
				for ( unsigned int k = 0; k < 6; k++ )
					vec( k ) = gsl_ran_gaussian( r, 1.0 );


				vec( 3 ) = vec( 4 ) = vec( 5 ) = 0.0;

				vec = ( chol.matrixL() * vec ).eval();

				vec = ( surfel.mean_ + vec ).eval();

				pcl::PointXYZRGB p;
				p.x = vec( 0 );
				p.y = vec( 1 );
				p.z = vec( 2 );

				const float L = vec( 3 );
				const float alpha = vec( 4 );
				const float beta = vec( 5 );

				float rf = 0, gf = 0, bf = 0;
				convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );

				int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
				int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
				int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );

				int rgb = ( r << 16 ) + ( g << 8 ) + b;
				p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

				info->cloudPtr->points.push_back( p );

			}

		}
		else {

			// PCA projection
			Eigen::Matrix< double, 6, 6 > cov = surfel.cov_;
			Eigen::Matrix< double, 3, 3 > cov3_inv = cov.block<3,3>(0,0).inverse();

			Eigen::Matrix< double, 3, 1> eigen_values_;
			Eigen::Matrix< double, 3, 3> eigen_vectors_;

			pcl::eigen33(Eigen::Matrix3d(cov.block<3,3>(0,0)), eigen_vectors_, eigen_values_);

			eigen_values_(0) = 0.0;
			eigen_values_(1) = sqrt( eigen_values_(1) );
			eigen_values_(2) = sqrt( eigen_values_(2) );

			Eigen::Matrix< double, 3, 3 > L = eigen_vectors_ * eigen_values_.asDiagonal();

			std::vector< Eigen::Matrix< double, 3, 1 >, Eigen::aligned_allocator< Eigen::Matrix< double, 3, 1 > > > vecs;

			Eigen::Matrix< double, 3, 1 > v;

			v(0) =  0.0; v(1) =  -1.0; v(2) =  -1.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  -1.0; v(2) =  0.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  -1.0; v(2) =  1.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  0.0; v(2) =  -1.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  0.0; v(2) =  0.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  0.0; v(2) =  1.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  1.0; v(2) =  -1.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  1.0; v(2) =  0.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  1.0; v(2) =  1.0;
			vecs.push_back( v );


			for( unsigned int k = 0; k < vecs.size(); k++ ) {

				Eigen::Matrix< double, 3, 1 > vec = 1.1*vecs[k];

				vec = ( L * vec ).eval();

				vec = ( surfel.mean_.block<3,1>(0,0) + vec ).eval();

				pcl::PointXYZRGB p;
				p.x = vec( 0 );
				p.y = vec( 1 );
				p.z = vec( 2 );

				// get color mean conditioned on position
				Eigen::Matrix< double, 3, 1 > cvec = surfel.mean_.block<3,1>(3,0) + cov.block<3,3>(3,0) * cov3_inv * ( vec - surfel.mean_.block<3,1>(0,0) );

				const float L = cvec( 0 );
				const float alpha = cvec( 1 );
				const float beta = cvec( 2 );

				float rf = 0, gf = 0, bf = 0;
				convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );

				int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
				int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
				int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );

				int rgb = ( r << 16 ) + ( g << 8 ) + b;
				p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

				info->cloudPtr->points.push_back( p );

			}

//			for( unsigned int j = 0; j < 2; j++ ) {
//				for( unsigned int k = 0; k < 2; k++ ) {
//					for( unsigned int l = 0; l < 2; l++ ) {
//
//						spatialaggregate::OcTreeNode< float, T >* neighbor = current->getNeighbor( j, k, l );
//
//						if( !neighbor )
//							continue;
//
////						if( neighbor->value_.surfels_[i].num_points_ < MIN_SURFEL_POINTS )
////							continue;
//
//						Surfel surfel;
//						surfel.num_points_ = current->value_.surfels_[i].num_points_ + neighbor->value_.surfels_[i].num_points_;
//						surfel.mean_ = current->value_.surfels_[i].num_points_ * current->value_.surfels_[i].mean_ + neighbor->value_.surfels_[i].num_points_ * neighbor->value_.surfels_[i].mean_;
//
//						surfel.mean_ /= surfel.num_points_;
//
//						pcl::PointXYZRGB p;
//						p.x = surfel.mean_( 0 );
//						p.y = surfel.mean_( 1 );
//						p.z = surfel.mean_( 2 );
//
//						const float L = surfel.mean_( 3 );
//						const float alpha = surfel.mean_( 4 );
//						const float beta = surfel.mean_( 5 );
//
//						float rf = 0, gf = 0, bf = 0;
//						convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );
//
//						int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
//						int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
//						int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );
//
//						int rgb = ( r << 16 ) + ( g << 8 ) + b;
//						p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );
//
//						info->cloudPtr->points.push_back( p );
//
//					}
//				}
//			}

		}
	}

}


struct Visualize3DColorDistributionWithNormalInfo {
	pcl::PointCloud< pcl::PointXYZRGBNormal >::Ptr cloudPtr;
	int viewDir, depth;
	bool random;
	int numSamples;
};

template <typename T>
void MultiResolutionColorSurfelMap<T>::visualize3DColorDistributionWithNormals( pcl::PointCloud< pcl::PointXYZRGBNormal >::Ptr cloudPtr, int depth, int viewDir, bool random, int numSamples ) {

	Visualize3DColorDistributionWithNormalInfo info;
	info.cloudPtr = cloudPtr;
	info.viewDir = viewDir;
	info.depth = depth;
	info.random = random;
	info.numSamples = numSamples;

	octree_->root_->sweepDown( &info, &visualize3DColorDistributionWithNormalsFunction );

}

template <typename T>
void MultiResolutionColorSurfelMap<T>::visualize3DColorDistributionWithNormalsFunction( spatialaggregate::OcTreeNode< float, T >* current, spatialaggregate::OcTreeNode< float, T >* next, void* data ) {

	Visualize3DColorDistributionWithNormalInfo* info = (Visualize3DColorDistributionWithNormalInfo*) data;

	if( (info->depth == -1 && current->type_ == spatialaggregate::OCTREE_BRANCHING_NODE) )
		return;

	if( info->depth >= 0 && current->depth_ != info->depth )
		return;

	Eigen::Matrix< float, 4, 1 > minPos = current->getMinPosition();
	Eigen::Matrix< float, 4, 1 > maxPos = current->getMaxPosition();


	// generate markers for histogram surfels
	for ( unsigned int i = 0; i < current->value_.numberOfSurfels; i++ ) {

		if( info->viewDir >= 0 && info->viewDir != i )
			continue;

		const MultiResolutionColorSurfelMap<T>::Surfel& surfel = current->value_.surfels_[i];

		if ( surfel.num_points_ < MIN_SURFEL_POINTS )
			continue;

		if( info->random ) {

			// samples N points from the normal distribution in mean and cov...
			unsigned int N = info->numSamples;

			// cholesky decomposition
			Eigen::Matrix< double, 6, 6 > cov = surfel.cov_;
			Eigen::LLT< Eigen::Matrix< double, 6, 6 > > chol = cov.llt();


			for ( unsigned int j = 0; j < N; j++ ) {

				Eigen::Matrix< double, 6, 1 > vec;
				for ( unsigned int k = 0; k < 6; k++ )
					vec( k ) = gsl_ran_gaussian( r, 1.0 );


				vec( 3 ) = vec( 4 ) = vec( 5 ) = 0.0;

				vec = ( chol.matrixL() * vec ).eval();

				vec = ( surfel.mean_ + vec ).eval();

				pcl::PointXYZRGBNormal p;
				p.x = vec( 0 );
				p.y = vec( 1 );
				p.z = vec( 2 );

				const float L = vec( 3 );
				const float alpha = vec( 4 );
				const float beta = vec( 5 );

				float rf = 0, gf = 0, bf = 0;
				convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );

				int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
				int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
				int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );

				int rgb = ( r << 16 ) + ( g << 8 ) + b;
				p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

				p.normal_x = surfel.normal_(0);
				p.normal_y = surfel.normal_(1);
				p.normal_z = surfel.normal_(2);

				info->cloudPtr->points.push_back( p );

			}

		}
		else {

			// PCA projection
			Eigen::Matrix< double, 6, 6 > cov = surfel.cov_;
			Eigen::Matrix< double, 3, 3 > cov3_inv = cov.block<3,3>(0,0).inverse();

			Eigen::Matrix< double, 3, 1> eigen_values_;
			Eigen::Matrix< double, 3, 3> eigen_vectors_;

			pcl::eigen33(Eigen::Matrix3d(cov.block<3,3>(0,0)), eigen_vectors_, eigen_values_);

			eigen_values_(0) = 0.0;
			eigen_values_(1) = sqrt( eigen_values_(1) );
			eigen_values_(2) = sqrt( eigen_values_(2) );

			Eigen::Matrix< double, 3, 3 > L = eigen_vectors_ * eigen_values_.asDiagonal();

			std::vector< Eigen::Matrix< double, 3, 1 >, Eigen::aligned_allocator< Eigen::Matrix< double, 3, 1 > > > vecs;

			Eigen::Matrix< double, 3, 1 > v;

			v(0) =  0.0; v(1) =  -1.0; v(2) =  -1.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  -1.0; v(2) =  0.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  -1.0; v(2) =  1.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  0.0; v(2) =  -1.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  0.0; v(2) =  0.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  0.0; v(2) =  1.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  1.0; v(2) =  -1.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  1.0; v(2) =  0.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  1.0; v(2) =  1.0;
			vecs.push_back( v );

			for( unsigned int k = 0; k < vecs.size(); k++ ) {

				Eigen::Matrix< double, 3, 1 > vec = 1.1*vecs[k];

				vec = ( L * vec ).eval();

				vec = ( surfel.mean_.block<3,1>(0,0) + vec ).eval();


				pcl::PointXYZRGBNormal p;
				p.x = vec( 0 );
				p.y = vec( 1 );
				p.z = vec( 2 );

				// get color mean conditioned on position
				Eigen::Matrix< double, 3, 1 > cvec = surfel.mean_.block<3,1>(3,0) + cov.block<3,3>(3,0) * cov3_inv * ( vec - surfel.mean_.block<3,1>(0,0) );

				const float L = cvec( 0 );
				const float alpha = cvec( 1 );
				const float beta = cvec( 2 );

				float rf = 0, gf = 0, bf = 0;
				convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );

				int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
				int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
				int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );

				int rgb = ( r << 16 ) + ( g << 8 ) + b;
				p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

				p.normal_x = surfel.normal_(0);
				p.normal_y = surfel.normal_(1);
				p.normal_z = surfel.normal_(2);

				info->cloudPtr->points.push_back( p );

			}

//			for( unsigned int j = 0; j < 2; j++ ) {
//				for( unsigned int k = 0; k < 2; k++ ) {
//					for( unsigned int l = 0; l < 2; l++ ) {
//
//						spatialaggregate::OcTreeNode< float, T >* neighbor = current->getNeighbor( j, k, l );
//
//						if( !neighbor )
//							continue;
//
////						if( neighbor->value_.surfels_[i].num_points_ < MIN_SURFEL_POINTS )
////							continue;
//
//						Surfel surfel;
//						surfel.num_points_ = current->value_.surfels_[i].num_points_ + neighbor->value_.surfels_[i].num_points_;
//						surfel.mean_ = current->value_.surfels_[i].num_points_ * current->value_.surfels_[i].mean_ + neighbor->value_.surfels_[i].num_points_ * neighbor->value_.surfels_[i].mean_;
//						surfel.cov_ = (current->value_.surfels_[i].num_points_-1.f) * current->value_.surfels_[i].cov_ + (neighbor->value_.surfels_[i].num_points_-1.f) * neighbor->value_.surfels_[i].cov_;
//
//						surfel.mean_ /= surfel.num_points_;
//						surfel.cov_ /= (surfel.num_points_-1.f);
//
//						surfel.evaluateNormal();
//
//						pcl::PointXYZRGBNormal p;
//						p.x = surfel.mean_( 0 );
//						p.y = surfel.mean_( 1 );
//						p.z = surfel.mean_( 2 );
//
//						const float L = surfel.mean_( 3 );
//						const float alpha = surfel.mean_( 4 );
//						const float beta = surfel.mean_( 5 );
//
//						float rf = 0, gf = 0, bf = 0;
//						convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );
//
//						int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
//						int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
//						int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );
//
//						int rgb = ( r << 16 ) + ( g << 8 ) + b;
//						p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );
//
//						p.normal_x = surfel.normal_(0);
//						p.normal_y = surfel.normal_(1);
//						p.normal_z = surfel.normal_(2);
//
//						info->cloudPtr->points.push_back( p );
//
//					}
//				}
//			}


		}
	}

}



struct Visualize3DColorMeansInfo {
	pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr;
	int viewDir, depth;
};

template <typename T>
void MultiResolutionColorSurfelMap<T>::visualize3DColorMeans( pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr, int depth, int viewDir ) {

	Visualize3DColorMeansInfo info;
	info.cloudPtr = cloudPtr;
	info.viewDir = viewDir;
	info.depth = depth;

	octree_->root_->sweepDown( &info, &visualizeMeansFunction );

}

template <typename T>
inline void MultiResolutionColorSurfelMap<T>::visualizeMeansFunction( spatialaggregate::OcTreeNode< float, T >* current, spatialaggregate::OcTreeNode< float, T >* next, void* data ) {

	Visualize3DColorMeansInfo* info = (Visualize3DColorMeansInfo*) data;

	if( (info->depth == -1 && current->type_ == spatialaggregate::OCTREE_BRANCHING_NODE) )
		return;

	if( info->depth >= 0 && current->depth_ != info->depth )
		return;

	Eigen::Matrix< float, 4, 1 > minPos = current->getMinPosition();
	Eigen::Matrix< float, 4, 1 > maxPos = current->getMaxPosition();


	for ( unsigned int i = 0; i < current->value_.numberOfSurfels; i++ ) {

		if( info->viewDir >= 0 && info->viewDir != i )
			continue;

		const MultiResolutionColorSurfelMap::Surfel& surfel = current->value_.surfels_[i];

		if ( surfel.num_points_ < MIN_SURFEL_POINTS )
			continue;

		pcl::PointXYZRGB p;
		p.x = surfel.mean_( 0 );
		p.y = surfel.mean_( 1 );
		p.z = surfel.mean_( 2 );

		const float L = surfel.mean_( 3 );
		const float alpha = surfel.mean_( 4 );
		const float beta = surfel.mean_( 5 );

		float rf = 0, gf = 0, bf = 0;
		convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );

		int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
		int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
		int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );

		int rgb = ( r << 16 ) + ( g << 8 ) + b;
		p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

		info->cloudPtr->points.push_back( p );

	}

}



struct VisualizeContoursInfo {
	pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr;
	Eigen::Matrix4d transform;
	int viewDir, depth;
	bool random;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

template <typename T>
void MultiResolutionColorSurfelMap<T>::visualizeContours( pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr, const Eigen::Matrix4d& transform, int depth, int viewDir, bool random ) {

	VisualizeContoursInfo info;
	info.transform = transform;
	info.cloudPtr = cloudPtr;
	info.viewDir = viewDir;
	info.depth = depth;
	info.random = random;

	octree_->root_->sweepDown( &info, &visualizeContoursFunction );

}

template <typename T>
inline void MultiResolutionColorSurfelMap<T>::visualizeContoursFunction( spatialaggregate::OcTreeNode< float, T >* current, spatialaggregate::OcTreeNode< float, T >* next, void* data ) {

	VisualizeContoursInfo* info = (VisualizeContoursInfo*) data;

	if( (info->depth == -1 && current->type_ != spatialaggregate::OCTREE_LEAF_NODE) )
		return;

	if( info->depth >= 0 && current->depth_ != info->depth )
		return;


	Eigen::Matrix< float, 4, 1 > minPos = current->getMinPosition();
	Eigen::Matrix< float, 4, 1 > maxPos = current->getMaxPosition();


	for ( unsigned int i = 0; i < 6; i++ ) {

		if( info->viewDir >= 0 && info->viewDir != i )
			continue;

		const MultiResolutionColorSurfelMap<T>::Surfel& surfel = current->value_.surfels_[i];

		if ( surfel.num_points_ < MIN_SURFEL_POINTS )
			continue;

		// determine angle between surfel normal and view direction onto surfel
		Eigen::Vector3d viewDirection = surfel.mean_.block<3,1>(0,0) - info->transform.block<3,1>(0,3);
		viewDirection.normalize();

		float cangle = viewDirection.dot( surfel.normal_ );

		// cholesky decomposition
		Eigen::Matrix< double, 6, 6 > cov = surfel.cov_;
		Eigen::LLT< Eigen::Matrix< double, 6, 6 > > chol = cov.llt();

		std::vector< Eigen::Matrix< double, 6, 1 >, Eigen::aligned_allocator< Eigen::Matrix< double, 6, 1 > > > vecs;

		Eigen::Matrix< double, 6, 1 > v;
		v.setZero();

		vecs.push_back( v );

		v(0) =  1.0; v(1) =  0.0; v(2) =  0.0;
		vecs.push_back( v );

		v(0) = -1.0; v(1) =  0.0; v(2) =  0.0;
		vecs.push_back( v );

		v(0) =  0.0; v(1) =  1.0; v(2) =  0.0;
		vecs.push_back( v );

		v(0) =  0.0; v(1) = -1.0; v(2) =  0.0;
		vecs.push_back( v );

		v(0) =  0.0; v(1) =  0.0; v(2) =  1.0;
		vecs.push_back( v );

		v(0) =  0.0; v(1) =  0.0; v(2) = -1.0;
		vecs.push_back( v );


		v(0) =  1.0; v(1) =  1.0; v(2) =  1.0;
		vecs.push_back( v );

		v(0) =  1.0; v(1) =  1.0; v(2) = -1.0;
		vecs.push_back( v );

		v(0) =  1.0; v(1) = -1.0; v(2) =  1.0;
		vecs.push_back( v );

		v(0) =  1.0; v(1) = -1.0; v(2) = -1.0;
		vecs.push_back( v );

		v(0) =  -1.0; v(1) =  1.0; v(2) =  1.0;
		vecs.push_back( v );

		v(0) =  -1.0; v(1) =  1.0; v(2) = -1.0;
		vecs.push_back( v );

		v(0) =  -1.0; v(1) = -1.0; v(2) =  1.0;
		vecs.push_back( v );

		v(0) =  -1.0; v(1) = -1.0; v(2) = -1.0;
		vecs.push_back( v );

		for( unsigned int k = 0; k < vecs.size(); k++ ) {

			Eigen::Matrix< double, 6, 1 > vec = 1.1*vecs[k];

			vec = ( chol.matrixL() * vec ).eval();

			vec = ( surfel.mean_ + vec ).eval();

			pcl::PointXYZRGB p;
			p.x = vec( 0 );
			p.y = vec( 1 );
			p.z = vec( 2 );

			const float L = vec( 3 );
			const float alpha = vec( 4 );
			const float beta = vec( 5 );

			float rf = 0, gf = 0, bf = 0;

			rf = 1.f;
			gf = fabsf( cangle );
			bf = fabsf( cangle );

//			convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );

			int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
			int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
			int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );

			int rgb = ( r << 16 ) + ( g << 8 ) + b;
			p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

//			if( fabsf( cangle ) > 0.2 )
//				continue;

			info->cloudPtr->points.push_back( p );

		}

	}

}



struct VisualizeNormalsInfo {
	pcl::PointCloud< pcl::PointXYZRGBNormal >::Ptr cloudPtr;
	int viewDir, depth;
};

template <typename T>
void MultiResolutionColorSurfelMap<T>::visualizeNormals( pcl::PointCloud< pcl::PointXYZRGBNormal >::Ptr cloudPtr, int depth, int viewDir ) {

	VisualizeNormalsInfo info;
	info.cloudPtr = cloudPtr;
	info.viewDir = viewDir;
	info.depth = depth;

	octree_->root_->sweepDown( &info, &visualizeNormalsFunction );

}

template <typename T>
inline void MultiResolutionColorSurfelMap<T>::visualizeNormalsFunction( spatialaggregate::OcTreeNode< float, T >* current, spatialaggregate::OcTreeNode< float, T >* next, void* data ) {

	VisualizeNormalsInfo* info = (VisualizeNormalsInfo*) data;

	if( (info->depth == -1 && current->type_ != spatialaggregate::OCTREE_LEAF_NODE) )
		return;

	if( info->depth >= 0 && current->depth_ != info->depth )
		return;

	for ( unsigned int i = 0; i < current->value_.numberOfSurfels; i++ ) {

		if( info->viewDir >= 0 && info->viewDir != i )
			continue;

		const MultiResolutionColorSurfelMap<T>::Surfel& surfel = current->value_.surfels_[i];

		if ( surfel.num_points_ < MIN_SURFEL_POINTS )
			continue;

		pcl::PointXYZRGBNormal p;

		p.x = surfel.mean_(0);
		p.y = surfel.mean_(1);
		p.z = surfel.mean_(2);

		const float L = surfel.mean_( 3 );
		const float alpha = surfel.mean_( 4 );
		const float beta = surfel.mean_( 5 );

		float rf = 0, gf = 0, bf = 0;
		convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );

		int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
		int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
		int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );

		int rgb = ( r << 16 ) + ( g << 8 ) + b;
		p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

		p.normal_x = surfel.normal_(0);
		p.normal_y = surfel.normal_(1);
		p.normal_z = surfel.normal_(2);

		info->cloudPtr->points.push_back( p );

	}

}


template <typename T>
struct VisualizeSimilarityInfo {
	pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr;
	int viewDir, depth;
	spatialaggregate::OcTreeNode< float, T >* referenceNode;
	bool simple;
};


template <typename T>
void MultiResolutionColorSurfelMap<T>::visualizeSimilarity( spatialaggregate::OcTreeNode< float, T >* referenceNode, pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr, int depth, int viewDir, bool simple ) {

	VisualizeSimilarityInfo<T> info;
	info.cloudPtr = cloudPtr;
	info.viewDir = viewDir;
	info.depth = depth;
	info.simple = simple;

	info.referenceNode = referenceNode;

	if( !info.referenceNode )
		return;

	if( info.referenceNode->depth_ != depth )
		return;

	octree_->root_->sweepDown( &info, &visualizeSimilarityFunction );

}

template <typename T>
inline void MultiResolutionColorSurfelMap<T>::visualizeSimilarityFunction( spatialaggregate::OcTreeNode< float, T >* current, spatialaggregate::OcTreeNode< float, T >* next, void* data ) {

	const float maxDist = 0.1f;

	VisualizeSimilarityInfo<T>* info = (VisualizeSimilarityInfo<T>*) data;

	if( current->depth_ != info->depth )
		return;

	// generate markers for histogram surfels
	float minDist = std::numeric_limits<float>::max();
	for ( unsigned int i = 0; i < current->value_.numberOfSurfels; i++ ) {

		if( info->viewDir >= 0 && info->viewDir != i )
			continue;

		MultiResolutionColorSurfelMap<T>::Surfel& surfel = info->referenceNode->value_.surfels_[i];

		if( surfel.num_points_ < MIN_SURFEL_POINTS )
			continue;

		MultiResolutionColorSurfelMap<T>::Surfel& surfel2 = current->value_.surfels_[i];

		if( surfel2.num_points_ < MIN_SURFEL_POINTS )
			continue;

		if( info->simple ) {
			ShapeTextureFeature f1 = surfel.simple_shape_texture_features_;
			ShapeTextureFeature f2 = surfel2.simple_shape_texture_features_;
			f1.shape_ /= f1.num_points_;
			f1.texture_ /= f1.num_points_;
			f2.shape_ /= f2.num_points_;
			f2.texture_ /= f2.num_points_;
			minDist = std::min( minDist, f1.distance( f2 ) );
		}
		else
			minDist = std::min( minDist, surfel.agglomerated_shape_texture_features_.distance( surfel2.agglomerated_shape_texture_features_ ) );

	}

	if( minDist == std::numeric_limits<float>::max() )
		return;

	Eigen::Vector4f pos = current->getCenterPosition();

	pcl::PointXYZRGB p;
	p.x = pos( 0 );
	p.y = pos( 1 );
	p.z = pos( 2 );

	int r = std::max( 0, std::min( 255, (int) ( 255.0 * minDist / maxDist ) ) );
	int g = 255 - std::max( 0, std::min( 255, (int) ( 255.0 * minDist / maxDist ) ) );
	int b = 255 - std::max( 0, std::min( 255, (int) ( 255.0 * minDist / maxDist ) ) );

	int rgb = ( r << 16 ) + ( g << 8 ) + b;
	p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

	info->cloudPtr->points.push_back( p );

}



struct VisualizeBordersInfo {
	pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr;
	int viewDir, depth;
	bool foreground;
};

template <typename T>
void MultiResolutionColorSurfelMap<T>::visualizeBorders( pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr, int depth, int viewDir, bool foreground ) {

	VisualizeBordersInfo info;
	info.cloudPtr = cloudPtr;
	info.viewDir = viewDir;
	info.depth = depth;
	info.foreground = foreground;

	octree_->root_->sweepDown( &info, &visualizeBordersFunction );

}

template <typename T>
inline void MultiResolutionColorSurfelMap<T>::visualizeBordersFunction( spatialaggregate::OcTreeNode< float, T >* current, spatialaggregate::OcTreeNode< float, T >* next, void* data ) {

	VisualizeBordersInfo* info = (VisualizeBordersInfo*) data;

	if( current->depth_ != info->depth )
		return;

	for ( unsigned int i = 0; i < current->value_.numberOfSurfels; i++ ) {

		if( info->viewDir >= 0 && info->viewDir != i )
			continue;

		MultiResolutionColorSurfelMap<T>::Surfel& surfel = current->value_.surfels_[i];

		if( surfel.num_points_ < MIN_SURFEL_POINTS )
			continue;

		Eigen::Vector4f pos = current->getCenterPosition();

		pcl::PointXYZRGB p;
		p.x = pos( 0 );
		p.y = pos( 1 );
		p.z = pos( 2 );

		int r = 255;
		int g = 0;
		int b = 0;

		if( info->foreground && current->value_.border_ ) {
			r = 0; g = 255; b = 255;
		}

		if( !info->foreground && !surfel.applyUpdate_ ) {
			r = 0; g = 255; b = 255;
		}

		int rgb = ( r << 16 ) + ( g << 8 ) + b;
		p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

		info->cloudPtr->points.push_back( p );

	}


}


template <typename T>
struct VisualizePrincipalSurfaceInfo {
	pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr;
	int viewDir, depth;
	MultiResolutionColorSurfelMap<T>* map;
	std::vector< Eigen::Vector3d, Eigen::aligned_allocator< Eigen::Vector3d > > samples;
};

template <typename T>
void MultiResolutionColorSurfelMap<T>::visualizePrincipalSurface( pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr, int depth, int viewDir ) {

	VisualizePrincipalSurfaceInfo<T> info;
	info.cloudPtr = cloudPtr;
	info.viewDir = viewDir;
	info.depth = depth;
	info.map = this;

	// 2N+1 sample points in each dimension
	int N = 5;
	for( int dx = -N; dx <= N; dx++ ) {
		for( int dy = -N; dy <= N; dy++ ) {
			info.samples.push_back( Eigen::Vector3d( (float)dx / (float)N * 0.5f, (float)dy / (float)N * 0.5f, 0 ) );
		}
	}

	octree_->root_->sweepDown( &info, &visualizePrincipalSurfaceFunction );

}

template <typename T>
inline void MultiResolutionColorSurfelMap<T>::visualizePrincipalSurfaceFunction( spatialaggregate::OcTreeNode< float, T >* current, spatialaggregate::OcTreeNode< float, T >* next, void* data ) {

	VisualizePrincipalSurfaceInfo<T>* info = (VisualizePrincipalSurfaceInfo<T>*) data;

	if( current->depth_ != info->depth )
		return;

	float resolution = current->resolution();


	for( unsigned int i = 0; i < current->value_.numberOfSurfels; i++ ) {

		if( info->viewDir >= 0 && info->viewDir != i )
			continue;

		const MultiResolutionColorSurfelMap<T>::Surfel& surfel = current->value_.surfels_[i];

		if ( surfel.num_points_ < MIN_SURFEL_POINTS )
			continue;

		// project samples to principal surface of GMM in local neighborhood using subspace constrained mean shift
		std::vector< spatialaggregate::OcTreeNode< float, T >* > neighbors;
		neighbors.reserve(27);
		current->getNeighbors( neighbors );

		std::vector< Eigen::Vector3d, Eigen::aligned_allocator< Eigen::Vector3d > > centerPositions;
		std::vector< MultiResolutionColorSurfelMap<T>::Surfel* > surfels;
		surfels.reserve(neighbors.size());
		for( unsigned int j = 0; j < neighbors.size(); j++ ) {

			if( !neighbors[j] )
				continue;

			if( neighbors[j]->value_.surfels_[i].num_points_ < MIN_SURFEL_POINTS )
				continue;
			else {

				// precalculate centerpos of neighbor node
				Eigen::Vector3d centerPosN = neighbors[j]->getCenterPosition().template block<3,1>(0,0).template cast<double>();

				surfels.push_back( &neighbors[j]->value_.surfels_[i] );
				centerPositions.push_back( centerPosN );
			}
		}

		Eigen::Vector3d centerPos = current->getCenterPosition().template block<3,1>(0,0).template cast<double>();

		Eigen::Vector3d meani = surfel.mean_.block<3,1>(0,0);
		Eigen::Matrix3d covi = surfel.cov_.block<3,3>(0,0);

		if( covi.determinant() <= std::numeric_limits<double>::epsilon() )
			continue;

		// eigen decompose covariance to find rotation onto principal plane
		// eigen vectors are stored in the columns in ascending order
		Eigen::Matrix3d eigenVectors;
		Eigen::Vector3d eigenValues;
		pcl::eigen33( covi, eigenVectors, eigenValues );

		Eigen::Matrix3d R_cov;
		R_cov.setZero();
		R_cov.block<3,1>(0,0) = eigenVectors.col(2);
		R_cov.block<3,1>(0,1) = eigenVectors.col(1);
		R_cov.block<3,1>(0,2) = eigenVectors.col(0);

		// include resolution scale
		R_cov *= 1.2f*resolution;

		for( unsigned int j = 0; j < info->samples.size(); j++ ) {
			Eigen::Vector3d sample = meani + R_cov * info->samples[j];

			if( info->map->projectOnPrincipalSurface( sample, surfels, centerPositions, resolution ) ) {

				// dont draw in other node volumes
				if( (centerPos - sample).maxCoeff() < 0.55f*resolution && (centerPos - sample).minCoeff() > -0.55f*resolution ) {

					// conditional mean color in GMM

					Eigen::Vector3d meanSum; meanSum.setZero();
					double weightSum = 0;

					for( unsigned int k = 0; k < surfels.size(); k++ ) {

						Eigen::Vector3d means = surfels[k]->mean_.block<3,1>(0,0);
						Eigen::Matrix3d covs = surfels[k]->cov_.block<3,3>(0,0);
						Eigen::Matrix3d covs_raw = covs;
						covs *= INTERPOLATION_COV_FACTOR;

						if( covs.determinant() <= std::numeric_limits<double>::epsilon() )
							continue;

						Eigen::Vector3d centerDiff = centerPositions[k] - sample;
						const double dx = resolution - fabsf(centerDiff(0));
						const double dy = resolution - fabsf(centerDiff(1));
						const double dz = resolution - fabsf(centerDiff(2));
						if( dx < 0 || dy < 0 || dz < 0 )
							continue;

						double weight = dx*dy*dz;

						Eigen::Matrix3d invcovs = covs.inverse();

						Eigen::Vector3d us = invcovs * (sample-means);
						double dist = exp( -0.5 * (sample-means).dot( us ) );

						double prob = 1.0 / sqrt( 8.0 * M_PI*M_PI*M_PI * covs.determinant() ) * dist;

						Eigen::Vector3d meanc = surfels[k]->mean_.block<3,1>(3,0);
						const Eigen::Matrix3d cov_cs = surfels[k]->cov_.block<3,3>(3,0);
						const Eigen::Vector3d mean_cond_cs = meanc + cov_cs * covs_raw.inverse() * (sample - means);

						meanSum += weight * prob * mean_cond_cs;
						weightSum += weight * prob;
					}

					if( weightSum > 0 ) {

						meanSum /= weightSum;

						pcl::PointXYZRGB p;
						p.x = sample( 0 );
						p.y = sample( 1 );
						p.z = sample( 2 );

						const float L = meanSum( 0 );
						const float alpha = meanSum( 1 );
						const float beta = meanSum( 2 );

						float rf = 0, gf = 0, bf = 0;
						convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );

						int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
						int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
						int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );

						int rgb = ( r << 16 ) + ( g << 8 ) + b;
						p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

						info->cloudPtr->points.push_back( p );

					}

				}

			}

		}

	}

}

template <typename T>
bool MultiResolutionColorSurfelMap<T>::projectOnPrincipalSurface( Eigen::Vector3d& sample, const std::vector< MultiResolutionColorSurfelMap::Surfel* >& neighbors, const std::vector< Eigen::Vector3d, Eigen::aligned_allocator< Eigen::Vector3d > >& centerPositions, double resolution ) {

	if( neighbors.size() == 0 )
		return false;

	Eigen::Vector3d x = sample;

	int maxIterations = 10;
	double epsilon = 1e-4;

	Eigen::Matrix3d covadd;
	covadd.setIdentity();

	int it = 0;
	while( it < maxIterations ) {

		// evaluate pdf and gradients at each mixture component
		double prob = 0;
		Eigen::Vector3d grad; grad.setZero();
		Eigen::Matrix3d hess; hess.setZero();
		Eigen::Matrix3d covSum; covSum.setZero();
		Eigen::Vector3d meanSum; meanSum.setZero();
		double weightSum = 0;

		double max_dist = 0.0;

		// use trilinear interpolation weights

		for( unsigned int i = 0; i < neighbors.size(); i++ ) {

			Eigen::Vector3d meani = neighbors[i]->mean_.block<3,1>(0,0);
			Eigen::Matrix3d covi = neighbors[i]->cov_.block<3,3>(0,0);

			covi *= INTERPOLATION_COV_FACTOR;

			if( covi.determinant() <= std::numeric_limits<double>::epsilon() )
				continue;

			Eigen::Vector3d centerDiff = centerPositions[i] - x;
			const double dx = resolution - fabsf(centerDiff(0));
			const double dy = resolution - fabsf(centerDiff(1));
			const double dz = resolution - fabsf(centerDiff(2));
			if( dx < 0 || dy < 0 || dz < 0 )
				continue;

			double weight = dx*dy*dz;

			Eigen::Matrix3d invcovi = covi.inverse();

			Eigen::Vector3d ui = invcovi * (x-meani);
			double dist = exp( -0.5 * (x-meani).dot( ui ) );
			double probi = 1.0 / sqrt( 8.0 * M_PI*M_PI*M_PI * covi.determinant() ) * dist;
			max_dist = std::max( max_dist, dist );

			prob += weight * probi;
			grad -= weight * probi * ui;
			hess += weight * probi * (ui * (ui.transpose()).eval() - invcovi);

			meanSum += weight * probi * invcovi * meani;
			covSum += weight * probi * invcovi;

			weightSum += weight;
		}

		if( isnan(weightSum ) ) {
			return false;
		}

		prob /= weightSum;
		grad /= weightSum;
		hess /= weightSum;

		if( prob > 1e-12 && ( it < maxIterations-1 || max_dist > 0.05 / INTERPOLATION_COV_FACTOR ) ) {

			Eigen::Vector3d mean = covSum.inverse() * meanSum;
			Eigen::Matrix3d invcov = -1.0 / prob * hess + 1.0 / (prob*prob) * grad * (grad.transpose()).eval();

			// eigen decomposition of invcov
			// eigen vectors are stored in the columns in ascending order
			Eigen::Matrix3d eigenVectors;
			Eigen::Vector3d eigenValues;
			pcl::eigen33( invcov, eigenVectors, eigenValues );

			Eigen::Matrix< double, 3, 3 > V_ortho;
			V_ortho.setZero();
			V_ortho.block<3,1>(0,0) = eigenVectors.col(2);

			x = (x + V_ortho * (V_ortho.transpose()).eval() * (mean-x).eval()).eval();

			// stopping criterion
			if( fabsf( grad.dot( V_ortho.transpose() * grad ) ) / ( grad.norm() * (V_ortho.transpose() * grad).norm() ) < epsilon )
				break;

		}
		else
			return false;

		it++;

	}

	sample = x;
	return true;


}



// s. http://people.cs.vt.edu/~kafura/cs2704/op.overloading2.html
template< typename T, int rows, int cols >
std::ostream& operator<<( std::ostream& os, Eigen::Matrix< T, rows, cols >& m ) {
	for ( unsigned int i = 0; i < rows; i++ ) {
		for ( unsigned int j = 0; j < cols; j++ ) {
			T d = m( i, j );
			os.write( (char*) &d, sizeof(T) );
		}
	}

	return os;
}

template< typename T, int rows, int cols >
std::istream& operator>>( std::istream& os, Eigen::Matrix< T, rows, cols >& m ) {
	for ( unsigned int i = 0; i < rows; i++ ) {
		for ( unsigned int j = 0; j < cols; j++ ) {
			T d;
			os.read( (char*) &d, sizeof(T) );
			m( i, j ) = d;
		}
	}

	return os;
}


std::ostream& operator<<( std::ostream& os, NodeValue& v ) {

	for ( int i = 0; i < v.numberOfSurfels; i++ ) {
		os << v.surfels_[i].initial_view_dir_;
		os << v.surfels_[i].first_view_dir_;
		os.write( (char*) &v.surfels_[i].first_view_inv_dist_, sizeof(float) );
		os.write( (char*) &v.surfels_[i].num_points_, sizeof(double) );
		os << v.surfels_[i].mean_;
		os << v.surfels_[i].normal_;
//		os.write( (char*) &v.surfels_[i].surface_curvature_, sizeof(double) );
//		os.write( (char*) &v.surfels_[i].color_curvature_, sizeof(double) );
//		os.write( (char*) &v.surfels_[i].curvature_, sizeof(double) );
		os << v.surfels_[i].cov_;
		os << v.surfels_[i].agglomerated_shape_texture_features_.shape_;
		os << v.surfels_[i].agglomerated_shape_texture_features_.texture_;
		os.write( (char*) &v.surfels_[i].agglomerated_shape_texture_features_.num_points_, sizeof(float) );

		os.write( (char*) &v.surfels_[i].up_to_date_, sizeof(bool) );
		os.write( (char*) &v.surfels_[i].applyUpdate_, sizeof(bool) );
		os.write( (char*) &v.surfels_[i].became_robust_, sizeof(bool) );
		os.write( (char*) &v.surfels_[i].idx_, sizeof(int) );


	}

	return os;
}


std::istream& operator>>( std::istream& os, NodeValue& v ) {

	for ( int i = 0; i < v.numberOfSurfels; i++ ) {
		os >> v.surfels_[i].initial_view_dir_;
		os >> v.surfels_[i].first_view_dir_;
		os.read( (char*) &v.surfels_[i].first_view_inv_dist_, sizeof(float) );
		os.read( (char*) &v.surfels_[i].num_points_, sizeof(double) );
		os >> v.surfels_[i].mean_;
		os >> v.surfels_[i].normal_;
//		os.read( (char*) &v.surfels_[i].surface_curvature_, sizeof(double) );
//		os.read( (char*) &v.surfels_[i].color_curvature_, sizeof(double) );
//		os.read( (char*) &v.surfels_[i].curvature_, sizeof(double) );
		os >> v.surfels_[i].cov_;
		os.read( (char*) &v.surfels_[i].up_to_date_, sizeof(bool) );
		os.read( (char*) &v.surfels_[i].applyUpdate_, sizeof(bool) );
		os.read( (char*) &v.surfels_[i].became_robust_, sizeof(bool) );
		os.read( (char*) &v.surfels_[i].idx_, sizeof(int) );

	}

	return os;
}


std::ostream& operator<<( std::ostream& os, spatialaggregate::OcTreeNode< float, NodeValue >& node ) {

	os.write( (char*) &node.depth_, sizeof(int) );
	os.write( (char*) &node.pos_key_.x_, sizeof(uint32_t) );
	os.write( (char*) &node.pos_key_.y_, sizeof(uint32_t) );
	os.write( (char*) &node.pos_key_.z_, sizeof(uint32_t) );
	os.write( (char*) &node.min_key_.x_, sizeof(uint32_t) );
	os.write( (char*) &node.min_key_.y_, sizeof(uint32_t) );
	os.write( (char*) &node.min_key_.z_, sizeof(uint32_t) );
	os.write( (char*) &node.max_key_.x_, sizeof(uint32_t) );
	os.write( (char*) &node.max_key_.y_, sizeof(uint32_t) );
	os.write( (char*) &node.max_key_.z_, sizeof(uint32_t) );
	os.write( (char*) &node.type_, sizeof(spatialaggregate::OcTreeNodeType) );

	for( unsigned int n = 0; n < 27; n++ )
		os.write( (char*) &node.neighbors_[n], sizeof(spatialaggregate::OcTreeNode< float, NodeValue>*) );

	for( unsigned int c = 0; c < 8; c++ )
		os.write( (char*) &node.children_[c], sizeof(spatialaggregate::OcTreeNode< float, NodeValue >*) );

	os.write( (char*) &node.parent_, sizeof(spatialaggregate::OcTreeNode< float, NodeValue >*) );

	os.write( (char*) &node.tree_, sizeof(spatialaggregate::OcTree< float, NodeValue>*) );


	os << node.value_;

	return os;
}


std::istream& operator>>( std::istream& os, spatialaggregate::OcTreeNode< float, NodeValue >& node ) {

	os.read( (char*) &node.depth_, sizeof(int) );
	os.read( (char*) &node.pos_key_.x_, sizeof(uint32_t) );
	os.read( (char*) &node.pos_key_.y_, sizeof(uint32_t) );
	os.read( (char*) &node.pos_key_.z_, sizeof(uint32_t) );
	os.read( (char*) &node.min_key_.x_, sizeof(uint32_t) );
	os.read( (char*) &node.min_key_.y_, sizeof(uint32_t) );
	os.read( (char*) &node.min_key_.z_, sizeof(uint32_t) );
	os.read( (char*) &node.max_key_.x_, sizeof(uint32_t) );
	os.read( (char*) &node.max_key_.y_, sizeof(uint32_t) );
	os.read( (char*) &node.max_key_.z_, sizeof(uint32_t) );
	os.read( (char*) &node.type_, sizeof(spatialaggregate::OcTreeNodeType) );
	os >> node.value_;

	return os;

}

template <typename T>
void MultiResolutionColorSurfelMap<T>::save( const std::string& filename ) {

	// create downsampling map for the target
	algorithm::OcTreeSamplingMap< float, T > samplingMap = algorithm::downsampleOcTree( *octree_, false, octree_->max_depth_ );

	std::ofstream outfile( filename.c_str(), std::ios::out | std::ios::binary );

	// header information
	outfile.write( (char*) &min_resolution_, sizeof(float) );
	outfile.write( (char*) &max_range_, sizeof(float) );

	outfile << reference_pose_;

	for ( int i = 0; i <= octree_->max_depth_; i++ ) {
		int numNodes = samplingMap[i].size();
		outfile.write( (char*) &numNodes, sizeof(int) );

		for ( typename std::list< spatialaggregate::OcTreeNode< float, T>* >::iterator it = samplingMap[i].begin(); it != samplingMap[i].end(); ++it ) {
			outfile << *(*it);
		}
	}

}


template <typename T>
void MultiResolutionColorSurfelMap<T>::load( const std::string& filename ) {

	std::ifstream infile( filename.c_str(), std::ios::in | std::ios::binary );

	if ( !infile.is_open() ) {
		std::cout << "could not open file " << filename.c_str() << "\n";
	}

	infile.read( (char*) &min_resolution_, sizeof(float) );
	infile.read( (char*) &max_range_, sizeof(float) );

	infile >> reference_pose_;

	octree_ = boost::shared_ptr< spatialaggregate::OcTree< float, T > >( new spatialaggregate::OcTree< float, T >( Eigen::Matrix< float, 4, 1 >( 0.f, 0.f, 0.f, 0.f ), min_resolution_, max_range_ ) );
	octree_->allocator_->deallocateNode( octree_->root_ );
	octree_->root_ = NULL;

	for ( int i = 0; i <= octree_->max_depth_; i++ ) {
		int numNodesOnDepth = 0;
		infile.read( (char*) &numNodesOnDepth, sizeof(int) );

		for ( int j = 0; j < numNodesOnDepth; j++ ) {

			spatialaggregate::OcTreeNode< float, T >* node = octree_->allocator_->allocateNode();
			octree_->acquire( node );

			infile >> ( *node );

			// insert octree node into the tree
			// start at root and traverse the tree until we find an empty leaf
			spatialaggregate::OcTreeNode< float, T >* n = octree_->root_;

			if ( !n ) {
				node->parent_ = NULL;
				octree_->root_ = node;
			} else {

				// search for parent
				spatialaggregate::OcTreeNode< float, T >* n2 = n;
				while ( n2 ) {
					n = n2;
					n2 = n->children_[n->getOctant( node->pos_key_ )];
				}

				// assert that found parent node has the correct depth
				if ( n->depth_ != node->depth_ - 1 || n->type_ != spatialaggregate::OCTREE_BRANCHING_NODE ) {
					std::cout << "MultiResolutionMap::load(): bad things happen\n";
				} else {
					n->children_[n->getOctant( node->pos_key_ )] = node;
					node->parent_ = n;
				}
			}

		}

	}

}

template <typename T>
void MultiResolutionColorSurfelMap<T>::indexNodesRecursive( spatialaggregate::OcTreeNode< float, T >* node, int minDepth, int maxDepth, bool includeBranchingNodes ) {

	if( node->depth_ >= minDepth && node->depth_ <= maxDepth && ( includeBranchingNodes || node->type_ != spatialaggregate::OCTREE_BRANCHING_NODE ) ) {
		node->value_.idx_ = indexedNodes_.size();
		indexedNodes_.push_back( node );
	}
	else
		node->value_.idx_ = -1;

	for( unsigned int i = 0; i < 8; i ++ ) {
		if( node->children_[i] ) {
			indexNodesRecursive( node->children_[i], minDepth, maxDepth, includeBranchingNodes );
		}
	}

}


template <typename T>
void MultiResolutionColorSurfelMap<T>::indexNodes( int minDepth, int maxDepth, bool includeBranchingNodes ) {

	indexedNodes_.clear();
	indexNodesRecursive( octree_->root_, minDepth, maxDepth, includeBranchingNodes );

}


template class MultiResolutionColorSurfelMap<NodeValue>;

