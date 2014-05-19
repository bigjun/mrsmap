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

#ifndef MULTIRESOLUTION_CSURFEL_MAP_H_
#define MULTIRESOLUTION_CSURFEL_MAP_H_

#include <Eigen/Core>
#include <Eigen/Eigen>

#include <vector>
#include <set>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/vector_average.h>

#include <octreelib/spatialaggregate/octree.h>

#include <mrsmap/map/grids.h>
#include <mrsmap/map/shapetexture_feature.h>

#include <gsl/gsl_rng.h>

#include <pcl/common/time.h>

#include <opencv2/opencv.hpp>

#include <tbb/tbb.h>
#include <tbb/concurrent_queue.h>

#include <flann/flann.h>


#define MAX_NUM_SURFELS 6

#define MIN_SURFEL_POINTS 10.0
#define MAX_SURFEL_POINTS 10000.0 //stop at this point count, since the sums may get numerically unstable
#define NUM_SURFEL_POINTS_ROBUST 6.0

#define PAIR_RELATION_NEIGHBORHOOD_UPLAYERS 3

#define INTERPOLATION_COV_FACTOR 20.0


typedef std::vector< Eigen::Matrix< double, 4, 1 >, Eigen::aligned_allocator< Eigen::Matrix< double, 4, 1 > > > DescriptorVec;



namespace mrsmap {

	class PointFeature {
	public:
		PointFeature() {}

		PointFeature( unsigned int descriptorLength ) {
			has_depth_ = false;
		}
		~PointFeature() {}

		Eigen::Vector2d image_pos_;
		Eigen::Matrix2d image_cov_, image_assoc_cov_; // pos covariance
		Eigen::Vector3d origin_;
		Eigen::Quaterniond orientation_;
		Eigen::Vector4d pos_; // (x,y,d) relative to given transform (origin and orientation)
		Eigen::Matrix4d cov_; //, assoc_cov_; // pos covariance

		Eigen::Vector3d invzpos_; // (x,y,1/d) relative to given transform (origin and orientation), in inverse depth parametrization
		Eigen::Matrix3d invzcov_, invzinvcov_; //, assoc_invzcov_; // pos covariance, in inverse depth parametrization

		bool has_depth_;

	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	};

	class ColorSurfel {
	public:
		ColorSurfel() {
			clear();
		}

		~ColorSurfel() {}

		inline void clear() {

			num_points_ = 0.0;
			mean_.setZero();
			cov_.setZero();

			surface_curvature_ = 0.0;

			up_to_date_ = false;
			applyUpdate_ = true;
			unevaluated_ = false;

			eff_view_dist_ = std::numeric_limits<float>::max();

			became_robust_ = false;

			assocWeight_ = 1.f;

			idx_ = -1;

		}


		inline ColorSurfel& operator+=(const ColorSurfel& rhs) {

			if( rhs.num_points_ > 0 && num_points_ < MAX_SURFEL_POINTS ) {

				// numerically stable one-pass update scheme
				if( num_points_ <= std::numeric_limits<double>::epsilon() ) {
					cov_ = rhs.cov_;
					mean_ = rhs.mean_;
					num_points_ = rhs.num_points_;
				}
				else {
					const Eigen::Matrix< double, 6, 1 > deltaS = rhs.num_points_ * mean_ - num_points_ * rhs.mean_;
					cov_ += rhs.cov_ + 1.0 / (num_points_ * rhs.num_points_ * (rhs.num_points_ + num_points_)) * deltaS * deltaS.transpose();
					mean_ += rhs.mean_;
					num_points_ += rhs.num_points_;
				}

				first_view_dir_ = rhs.first_view_dir_;
				first_view_inv_dist_ = rhs.first_view_inv_dist_;
				up_to_date_ = false;
			}

			return *this;
		}


		inline void add( const Eigen::Matrix< double, 6, 1 >& point ) {
			// numerically stable one-pass update scheme
			if( num_points_ < std::numeric_limits<double>::epsilon() ) {
				mean_ += point;
				num_points_ += 1.0;
				up_to_date_ = false;
			}
			else if( num_points_ < MAX_SURFEL_POINTS ) {
				const Eigen::Matrix< double, 6, 1 > deltaS = (mean_ - num_points_ * point);
				cov_ += 1.0 / (num_points_ * (num_points_ + 1.0)) * deltaS * deltaS.transpose();
				mean_ += point;
				num_points_ += 1.0;
				up_to_date_ = false;
			}
		}


//		inline void add( const Eigen::Matrix< double, 6, 1 >& sum, const Eigen::Matrix< double, 6, 6 >& sum_squares, double num_points ) {
//			if( num_points > 0 && num_points_ < MAX_SURFEL_POINTS ) {
//				mean_ += sum;
//				cov_ += sum_squares;
//				num_points_ += num_points;
//				up_to_date_ = false;
//			}
//		}

		inline void evaluateNormal() {

			Eigen::Matrix< double, 3, 1> eigen_values_;
			Eigen::Matrix< double, 3, 3> eigen_vectors_;

			// eigen vectors are stored in the columns
			pcl::eigen33(Eigen::Matrix3d(cov_.block<3,3>(0,0)), eigen_vectors_, eigen_values_);

			normal_ = eigen_vectors_.col(0);
			if( normal_.dot( first_view_dir_ ) > 0.0 )
				normal_ *= -1.0;

//			double sumEigVals = eigen_values_.sum();
//
//			if( sumEigVals > 1e-10 )
//				surface_curvature_ = eigen_values_.minCoeff() / sumEigVals;
//			else
//				surface_curvature_ = 0.0;


		}

		inline void evaluate() {

			// determine effective view distance
			eff_view_dist_ = first_view_dir_.dot( initial_view_dir_ ) * first_view_inv_dist_;

			if( num_points_ >= MIN_SURFEL_POINTS ) {

				const double inv_num = 1.0 / num_points_;
				mean_ *= inv_num;
				cov_ /= (num_points_-1.0);
//				cov_ = (inv_num * cov_ - mean_ * mean_.transpose()).eval();


				// enforce symmetry..
				cov_(1,0) = cov_(0,1);
				cov_(2,0) = cov_(0,2);
				cov_(3,0) = cov_(0,3);
				cov_(4,0) = cov_(0,4);
				cov_(5,0) = cov_(0,5);
				cov_(2,1) = cov_(1,2);
				cov_(2,3) = cov_(3,2);
				cov_(2,4) = cov_(4,2);
				cov_(2,5) = cov_(5,2);
				cov_(3,1) = cov_(1,3);
				cov_(3,4) = cov_(4,3);
				cov_(3,5) = cov_(5,3);
				cov_(4,1) = cov_(1,4);
				cov_(4,5) = cov_(5,4);
				cov_(5,1) = cov_(1,5);

				double det = cov_.block<3,3>(0,0).determinant();

				if( det <= std::numeric_limits<double>::epsilon() ) {
//					cov_(0,0) += 0.00001;
//					cov_(1,1) += 0.00001;
//					cov_(2,2) += 0.00001;

					mean_.setZero();
					cov_.setZero();

					num_points_ = 0;
				}
//				else {

//					Eigen::Matrix< double, 3, 1> eigen_values_;
//					Eigen::Matrix< double, 3, 3> eigen_vectors_;
//
//					// eigen vectors are stored in the columns
//					pcl::eigen33(Eigen::Matrix3d(cov_.block<3,3>(0,0)), eigen_vectors_, eigen_values_);
//
//					normal_ = eigen_vectors_.col(0);
//					if( normal_.dot( first_view_dir_ ) > 0.0 )
//						normal_ *= -1.0;

	//						double sumEigVals = eigen_values_.sum();
	//
	//						if( sumEigVals > 1e-10 )
	//							surface_curvature_ = eigen_values_(0) / sumEigVals;
	//						else
	//							surface_curvature_ = 0.0;
	//
	//						color_curvature_ = cov_(3,3) + cov_(4,4) + cov_(5,5);
	//
	//						curvature_ = std::min( 1.0, std::max( 0.0, (10.0*surface_curvature_ + 9.0*color_curvature_) ) );

//					cov_add_.setZero();

//				}

			}

			up_to_date_ = true;
			unevaluated_ = false;

		}


		inline void unevaluate() {

			if( num_points_ > 0.0 ) {

				mean_ *= num_points_;
				cov_ *= (num_points_-1.0);

				unevaluated_ = true;

			}

//			up_to_date_ = false;

		}


	  Eigen::Matrix< double, 3, 1 > initial_view_dir_, first_view_dir_;

	  float first_view_inv_dist_;
	  float eff_view_dist_;
	  double num_points_;

	  Eigen::Matrix< double, 6, 1 > mean_;
	  Eigen::Matrix< double, 3, 1 > normal_;
	  double surface_curvature_;
	//		  double color_curvature_;
	//		  double curvature_;
	  Eigen::Matrix< double, 6, 6 > cov_;
//	  Eigen::Matrix< double, 3, 3 > cov_add_; // depends on flatness..
	  bool up_to_date_, applyUpdate_;
	  bool unevaluated_;

	  bool became_robust_; // is set to true, when the point count passes NUM_SURFEL_POINTS_ROBUST
	  int idx_;

	  float assocDist_;
	  float assocWeight_;

	  ShapeTextureFeature simple_shape_texture_features_;
	  ShapeTextureFeature agglomerated_shape_texture_features_;

	public:
	  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	};


	class MultiResolutionColorSurfelMap
	{
	public:


		class Params {
		public:
			Params();

			~Params() {}

			float	depthNoiseFactor;
			float	pixelNoise;

			float	depthNoiseAssocFactor;
			float	pixelNoiseAssocFactor;

			bool	usePointFeatures;
			bool	debugPointFeatures;

			unsigned int	GridCols;
			unsigned int	GridRows;
			unsigned int	GridCellMax;
			unsigned int	numPointFeatures;

			double dist_dependency;

		};



		typedef ColorSurfel Surfel;

		class ContextSignatureInformation {
		public:
			ContextSignatureInformation() {}

			ContextSignatureInformation( double max_distance, const Eigen::Vector4d& centerRadii, const Eigen::Vector4d& margins )
			: grid_( SIGNATURE_ANGLES, SIGNATURE_RADII, max_distance, centerRadii, margins ) {}
			~ContextSignatureInformation() {}

			Eigen::Vector3d refAxis1, refAxis2;
			ThreeLayeredPolarGrid2D< 4 > grid_;

		public:
				EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		};


		class ContextSignature {
		public:
			ContextSignature() {}
			~ContextSignature() {}

			inline int hammingDistance( const ContextSignature& rhs ) const {

				return (context_signature_ ^ rhs.context_signature_).count();

			}

			/*
			 * computes s ^ (m | s), i.e. only couts missing bits towards the model (m = rhs)
			 */
			inline int hammingDistanceToModel( const ContextSignature& rhs ) const {

				return ( context_signature_ ^ ( rhs.context_signature_ | context_signature_ ) ).count();

			}

			std::bitset< SIGNATURE_ANGLES*SIGNATURE_RADII*SIGNATURE_ELEMENTS > context_signature_;

		};



		class SurfelPairSignature {
		public:
			SurfelPairSignature() {
			}

			~SurfelPairSignature() {
			}

			// distance, dot products of n1 to d, n2 to d, and n1 to n2
			Eigen::Matrix< double, 4, 1 > shape_signature_;

			// distance in L, alpha, beta color space
			Eigen::Matrix< double, 3, 1 > color_signature_;

			// hash key
			inline uint64 getKey() {

				// each signature value gets 1 Byte in the key
				const int bins_dist = 4;
				const int bins_angles = 30;

				uint64 bin_s1 = std::max( 0, std::min( bins_dist-1, (int)(shape_signature_(0) * ((double)bins_dist)) ) );
				uint64 bin_s2 = std::max( 0, std::min( bins_angles-1, (int)(shape_signature_(1) * ((double)bins_angles)) ) );
				uint64 bin_s3 = std::max( 0, std::min( bins_angles-1, (int)(shape_signature_(2) * ((double)bins_angles)) ) );
				uint64 bin_s4 = std::max( 0, std::min( bins_angles-1, (int)(shape_signature_(3) * ((double)bins_angles)) ) );

				const double lumHighThreshold = 0.5 + LUMINANCE_BIN_THRESHOLD;
				const double lumLowThreshold = 0.5 - LUMINANCE_BIN_THRESHOLD;
				const double colorHighThreshold = 0.5 + COLOR_BIN_THRESHOLD;
				const double colorLowThreshold = 0.5 - COLOR_BIN_THRESHOLD;

				uint64 bin_c1 = 0;
				if( color_signature_(0) > lumHighThreshold ) bin_c1 = 1;
				else if( color_signature_(0) < lumLowThreshold ) bin_c1 = 2;

				uint64 bin_c2 = 0;
				if( color_signature_(1) > colorHighThreshold ) bin_c2 = 1;
				else if( color_signature_(1) < colorLowThreshold ) bin_c2 = 2;

				uint64 bin_c3 = 0;
				if( color_signature_(2) > colorHighThreshold ) bin_c3 = 1;
				else if( color_signature_(2) < colorLowThreshold ) bin_c3 = 2;

				uint64 key = 0;
				key |= bin_s1;
				key |= bin_s2 << 8;
				key |= bin_s3 << 16;
				key |= bin_s4 << 24;
				key |= bin_c1 << 32;
				key |= bin_c2 << 40;
				key |= bin_c3 << 48;

				return key;

			}

			// position + quaternion, trafo from reference frame to map frame
			Eigen::Matrix< double, 7, 1 > reference_pose_;

			ContextSignature context_signature_;

		public:
				EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		};


		class SurfelPair {
		public:
			SurfelPair( Surfel* src, Surfel* dst, Eigen::Matrix< double, 7, 1 > referencePose, const ContextSignature& contextSignature )
			: src_(src), dst_(dst), reference_pose_( referencePose ), context_signature_( contextSignature ) {

			}

			~SurfelPair() {
			}

			Surfel* src_;
			Surfel* dst_;

			// position + quaternion, trafo from reference frame to map frame
			Eigen::Matrix< double, 7, 1 > reference_pose_;

			ContextSignature context_signature_;

		public:
				EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		};

		class NodeValue {
		public:
			NodeValue() {
				initialize();
			}

			NodeValue( unsigned int v ) {
				initialize();
			}

			~NodeValue() {}

			inline void initialize() {

				idx_ = -1;
				associated_ = 0;
				assocWeight_ = 1.f;
				border_ = false;

				surfels_[0].initial_view_dir_ = Eigen::Vector3d( 1., 0., 0. );
				surfels_[1].initial_view_dir_ = Eigen::Vector3d( -1., 0., 0. );
				surfels_[2].initial_view_dir_ = Eigen::Vector3d( 0., 1., 0. );
				surfels_[3].initial_view_dir_ = Eigen::Vector3d( 0., -1., 0. );
				surfels_[4].initial_view_dir_ = Eigen::Vector3d( 0., 0., 1. );
				surfels_[5].initial_view_dir_ = Eigen::Vector3d( 0., 0., -1. );

				for( unsigned int i = 0; i < 6; i++ )
					surfels_[i].first_view_dir_ = surfels_[i].initial_view_dir_;

			}


			inline NodeValue& operator+=(const NodeValue& rhs) {

				// merge surfels
				for( unsigned int i = 0; i < 6; i++ ) {

					Surfel& surfel = surfels_[i];

					if( surfel.applyUpdate_ ) {
						if( surfel.up_to_date_ )
							surfel.clear();

						surfel += rhs.surfels_[i];
					}

				}

				return *this;
			}


			inline Surfel* getSurfel( const Eigen::Vector3d& viewDirection ) {

				Surfel* bestMatchSurfel = NULL;
				double bestMatchDist = -1.;

				for( unsigned int i = 0; i < MAX_NUM_SURFELS; i++ ) {
					const double dist = viewDirection.dot( surfels_[i].initial_view_dir_ );
					if( dist > bestMatchDist ) {
						bestMatchSurfel = &surfels_[i];
						bestMatchDist = dist;
					}
				}

				return bestMatchSurfel;
			}


			inline void addSurfel( const Eigen::Vector3d& viewDirection, const Surfel& surfel ) {

				// find best matching surfel for the view direction
				Surfel* bestMatchSurfel = getSurfel( viewDirection );

				if( bestMatchSurfel->applyUpdate_ ) {
					if( bestMatchSurfel->up_to_date_ )
						bestMatchSurfel->clear();

					*bestMatchSurfel += surfel;
				}

			}


			inline void evaluateNormals( spatialaggregate::OcTreeNode<float, NodeValue>* node ) {
				for( unsigned int i = 0; i < 6; i++ ) {
					if( !surfels_[i].up_to_date_ || surfels_[i].unevaluated_ ) {

						Surfel surfel = surfels_[i];
						for( unsigned int n = 0; n < 27; n++ ) {
							if(node->neighbors_[n] && node->neighbors_[n] != node ) {
								surfel += node->neighbors_[n]->value_.surfels_[i];
							}
						}

						surfel.first_view_dir_ = surfels_[i].first_view_dir_;
						surfel.evaluate();
						surfel.evaluateNormal();

						surfels_[i].normal_ = surfel.normal_;

					}
				}
			}


			inline void evaluateSurfels() {
//				max_curvature_ = 0.0;
				for( unsigned int i = 0; i < 6; i++ ) {
					if( !surfels_[i].up_to_date_ || surfels_[i].unevaluated_ ) {
						surfels_[i].evaluate();
//						if( surfels_[i].num_points_ > MIN_SURFEL_POINTS )
//							max_curvature_ = std::max( max_curvature_, surfels_[i].curvature_ );
					}
				}
			}

			inline void unevaluateSurfels() {
//				max_curvature_ = 0.0;
				for( unsigned int i = 0; i < 6; i++ ) {
					if( surfels_[i].up_to_date_ ) {
						surfels_[i].unevaluate();
//						if( surfels_[i].num_points_ > MIN_SURFEL_POINTS )
//							max_curvature_ = std::max( max_curvature_, surfels_[i].curvature_ );
					}
				}
			}

			Surfel surfels_[6];
			char associated_; // -1: disabled, 0: not associated, 1: associated, 2: not associated but neighbor of associated node
			spatialaggregate::OcTreeNode<float, NodeValue>* association_;
			char assocSurfelIdx_, assocSurfelDstIdx_;
			float assocWeight_;

			int idx_;

			bool border_;

//			double max_curvature_;

			EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		};


		class ImagePreAllocator {
		public:
			ImagePreAllocator();

			~ImagePreAllocator();

			struct Info {
				Info() : value( NULL ) {}
				Info( NodeValue* v, const spatialaggregate::OcTreeKey< float, NodeValue >& k, unsigned int d )
				: value(v), key(k), depth(d) {}

				NodeValue* value;
				spatialaggregate::OcTreeKey< float, NodeValue > key;
				unsigned int depth;
			};

			void prepare( unsigned int w, unsigned int h, bool buildNodeImage );

			spatialaggregate::DynamicAllocator< NodeValue > imageNodeAllocator_;
			uint64_t* imgKeys;
			NodeValue** valueMap;
			std::vector< Info > infoList;
			tbb::concurrent_vector< std::vector< Info > > parallelInfoList;
			unsigned int width, height;
			spatialaggregate::OcTreeNode<float, NodeValue>** node_image_;
			std::set< spatialaggregate::OcTreeNode<float, NodeValue>* > node_set_;

			boost::mutex mutex_;

		};


		MultiResolutionColorSurfelMap( float minResolution, float radius, boost::shared_ptr< spatialaggregate::OcTreeNodeAllocator< float, NodeValue > > allocator = boost::make_shared< spatialaggregate::OcTreeNodeAllocator< float, NodeValue > >() );

		~MultiResolutionColorSurfelMap();


		void extents( Eigen::Matrix< double, 3, 1 >& mean, Eigen::Matrix< double, 3, 3 >& cov );


		struct NodeSurfel {
			spatialaggregate::OcTreeNode< float, NodeValue >* node;
			Surfel* surfel;
		};


		void addPoints( const boost::shared_ptr< const pcl::PointCloud<pcl::PointXYZRGB> >& cloud, const boost::shared_ptr< const std::vector< int > >& indices );

		void addPoints( const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const std::vector< int >& indices );

		void addImage( const pcl::PointCloud<pcl::PointXYZRGB>& cloud, bool smoothViewDir = true, bool buildNodeImage = false );

		class ImageAddFunctor {
		public:
			ImageAddFunctor( MultiResolutionColorSurfelMap* map, const pcl::PointCloud< pcl::PointXYZRGB >& cloud, bool smoothViewDir, bool buildNodeImage, boost::shared_ptr< ImagePreAllocator > imageAllocator );

			void operator()( const tbb::blocked_range<size_t>& r ) const;

			boost::shared_ptr< ImagePreAllocator > imageAllocator_;
			const pcl::PointCloud< pcl::PointXYZRGB >& cloud_;
			MultiResolutionColorSurfelMap* map_;
			bool smoothViewDir_;
		};

		void addImageParallel( const pcl::PointCloud<pcl::PointXYZRGB>& cloud, bool smoothViewDir = true, bool buildNodeImage = false );

		void addDisplacementImage( const pcl::PointCloud<pcl::PointXYZRGB>& cloud_pos,  const pcl::PointCloud<pcl::PointXYZRGB>& cloud_disp, bool smoothViewDir = true, bool buildNodeImage = false );

		void addImagePointFeatures( const cv::Mat& img, const pcl::PointCloud< pcl::PointXYZRGB >& cloud );

		void getImage( cv::Mat& img, const Eigen::Vector3d& viewPosition );

		static inline bool splitCriterion( spatialaggregate::OcTreeNode< float, NodeValue >* oldLeaf, spatialaggregate::OcTreeNode< float, NodeValue >* newLeaf );

		void findImageBorderPoints( const pcl::PointCloud<pcl::PointXYZRGB>& cloud, std::vector< int >& indices );

		void findVirtualBorderPoints( const pcl::PointCloud<pcl::PointXYZRGB>& cloud, std::vector< int >& indices );

		void findForegroundBorderPoints( const pcl::PointCloud<pcl::PointXYZRGB>& cloud, std::vector< int >& indices );

		void findContourPoints( const pcl::PointCloud<pcl::PointXYZRGB>& cloud, std::vector< int >& indices );

		void clearAtPoints( const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const std::vector< int >& indices );

		void markNoUpdateAtPoints( const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const std::vector< int >& indices );

		void clearUpdateSurfelsAtPoints( const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const std::vector< int >& indices );

		void markBorderAtPoints( const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const std::vector< int >& indices );

		void markBorderFromViewpoint( const Eigen::Vector3d& viewpoint );

		static inline void clearBorderFlagFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data );
		void clearBorderFlag();

		void markUpdateAllSurfels();
		static inline void markUpdateAllSurfelsFunction( spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data );

		void markUpdateImprovedEffViewDistSurfels( const Eigen::Vector3f& viewPosition );
		static inline void markUpdateImprovedEffViewDistSurfelsFunction( spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data );

		void evaluateSurfels();
		void unevaluateSurfels();

		bool pointInForeground( const Eigen::Vector3f& position, const cv::Mat& image_depth, const cv::Point2f imagePoint, float scale, float jumpThreshold );

		void setApplyUpdate( bool v );

		void setUpToDate( bool v );

		void clearUnstableSurfels();

		void clearRobustChangeFlag();

		void setRobustChangeFlag();

		void evaluateSurfelPairRelations();

		ContextSignature buildContextSignature( const Surfel& src, const Surfel& dst, const ContextSignatureInformation& csi, double max_distance, double resolution, int fileIdx = -1 );
		int traverseAndBuildSurfelPairRelationsOnDepth( spatialaggregate::OcTreeNode< float, NodeValue >* src, unsigned int surfelIdx, const ContextSignatureInformation& csi, std::list< spatialaggregate::OcTreeNode< float, NodeValue >* > dst, unsigned int depth );
		inline void prepareContextSignatureInformation( ContextSignatureInformation& csi, spatialaggregate::OcTreeNode< float, NodeValue >* src, unsigned int surfelIdx, std::list< spatialaggregate::OcTreeNode< float, NodeValue >* > dstNodes, unsigned int depth );

		inline SurfelPairSignature buildSurfelPairRelation( const Surfel& src, const Surfel& dst, const ContextSignatureInformation& csi, double maxDistance, double resolution );


		void buildShapeTextureFeatures();


		void clearAssociatedFlag();
		void distributeAssociatedFlag();

		void clearAssociationDist();

		void clearAssociations();
		static inline void clearAssociationsFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);


		std::vector< unsigned int > findInliers( const std::vector< unsigned int >& indices, const pcl::PointCloud<pcl::PointXYZRGB>& cloud, int maxDepth );

		void visualize3DColorDistribution( pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr, int depth, int viewDir, bool random = true );
		void visualize3DColorDistributionWithNormals( pcl::PointCloud< pcl::PointXYZRGBNormal >::Ptr cloudPtr, int depth, int viewDir, bool random = true, int numSamples = 100 );
		void visualize3DColorMeans( pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr, int depth, int viewDir );

		void visualizeContours( pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr, const Eigen::Matrix4d& transform, int depth, int viewDir, bool random = true );

		void visualizeSimilarity( spatialaggregate::OcTreeNode< float, NodeValue >* referenceNode, pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr, int depth, int viewDir, bool simple );

		void visualizeBorders( pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr, int depth, int viewDir, bool foreground );

		void visualizePrincipalSurface( pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr, int depth, int viewDir );
		bool projectOnPrincipalSurface( Eigen::Vector3d& sample, const std::vector< Surfel* >& neighbors, const std::vector< Eigen::Vector3d, Eigen::aligned_allocator< Eigen::Vector3d > >& centerPositions, double resolution );

		void visualizeNormals( pcl::PointCloud< pcl::PointXYZRGBNormal >::Ptr cloudPtr, int depth, int viewDir );


		static inline void evaluateNormalsFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void evaluateSurfelsFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void unevaluateSurfelsFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void clearUnstableSurfelsFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void setApplyUpdateFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void setUpToDateFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void clearRobustChangeFlagFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void setRobustChangeFlagFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void evaluateSurfelPairRelationsFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void clearAssociatedFlagFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void distributeAssociatedFlagFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void clearAssociationDistFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void buildSimpleShapeTextureFeatureFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void buildAgglomeratedShapeTextureFeatureFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void visualize3DColorDistributionFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void visualize3DColorDistributionWithNormalsFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void visualizeContoursFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void visualizeSimilarityFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void visualizeBordersFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void visualizePrincipalSurfaceFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void visualizeNormalsFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void visualizeMeansFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);

		void save( const std::string& filename );
		void load( const std::string& filename );


		void indexNodes( int minDepth, int maxDepth, bool includeBranchingNodes = true );
		void indexNodesRecursive( spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* node, int minDepth, int maxDepth, bool includeBranchingNodes );


		boost::shared_ptr< spatialaggregate::OcTree<float, NodeValue> > octree_;
		boost::shared_ptr< spatialaggregate::OcTreeNodeAllocator< float, NodeValue > > allocator_;
		boost::shared_ptr< ImagePreAllocator > imageAllocator_;

		Eigen::Matrix4d reference_pose_;


		float min_resolution_, max_range_;

		int last_pair_surfel_idx_;

		static gsl_rng* r;

//		std_msgs::Header header;

		std::map< unsigned int, std::map< int64, std::vector< SurfelPair, Eigen::aligned_allocator< SurfelPair > > > > surfel_pair_map_;

		pcl::StopWatch stopwatch_;

		std::vector< spatialaggregate::OcTreeNode<float, NodeValue>* > indexedNodes_;

		std::vector< PointFeature, Eigen::aligned_allocator< PointFeature > > features_;
		cv::Mat descriptors_;
		boost::shared_ptr< flann::Index< flann::HammingPopcnt< unsigned char > > > lsh_index_;

		Params params_;

		cv::Mat img_rgb_;

	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	};


};

#endif /* MULTIRESOLUTION_CSURFEL_MAP_H_ */

