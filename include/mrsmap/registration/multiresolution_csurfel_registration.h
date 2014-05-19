/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Computer Science Institute VI, University of Bonn
 *  Author: Joerg Stueckler, 16.05.2011
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


#ifndef MULTIRESOLUTION_CSURFEL_REGISTRATION_H_
#define MULTIRESOLUTION_CSURFEL_REGISTRATION_H_

#include <gsl/gsl_multimin.h>

#include "mrsmap/map/multiresolution_csurfel_map.h"

#include "octreelib/algorithm/downsample.h"

#include <list>


// takes in two map for which it estimates the rigid transformation with a coarse-to-fine strategy.
namespace mrsmap {

	class MultiResolutionColorSurfelRegistration {
	public:

		class Params {
		public:
			Params();
			~Params() {}

			void init();
			std::string toString();

			bool registerSurfels_, registerFeatures_;

			bool use_prior_pose_;
			Eigen::Matrix< double, 6, 1 > prior_pose_mean_;
			Eigen::Matrix< double, 6, 6 > prior_pose_invcov_;

			bool add_smooth_pos_covariance_;
			float smooth_surface_cov_factor_;
			double surfel_match_angle_threshold_;
			unsigned int registration_min_num_surfels_;
			double max_feature_dist2_;
			bool use_features_;
			bool match_likelihood_use_color_, registration_use_color_;
			double luminance_damp_diff_, luminance_reg_threshold_;
			double color_damp_diff_, color_reg_threshold_;
			double occlusion_z_similarity_factor_;
			unsigned int image_border_range_;

			bool parallel_;


			float startResolution_;
			float stopResolution_;

			int pointFeatureMatchingNumNeighbors_;
			int pointFeatureMatchingThreshold_;
			float pointFeatureMatchingCoarseImagePosMahalDist_, pointFeatureMatchingFineImagePosMahalDist_;
			float pointFeatureWeight_;
			double calibration_f_, calibration_c1_, calibration_c2_;
			Eigen::Matrix3d K_, KInv_;
			bool debugFeatures_;


		};


		MultiResolutionColorSurfelRegistration();
		MultiResolutionColorSurfelRegistration( const Params& params );
		~MultiResolutionColorSurfelRegistration() {}

		class SurfelAssociation {
		public:
			SurfelAssociation()
			: n_src_(NULL), src_(NULL), src_idx_(0), n_dst_(NULL), dst_(NULL), dst_idx_(0), match(0) {}
			SurfelAssociation( spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_src, MultiResolutionColorSurfelMap::Surfel* src, unsigned int src_idx, spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_dst, MultiResolutionColorSurfelMap::Surfel* dst, unsigned int dst_idx )
			: n_src_(n_src), src_(src), src_idx_(src_idx), n_dst_(n_dst), dst_(dst), dst_idx_(dst_idx), match(1) {}
			~SurfelAssociation() {}

			spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_src_;
			MultiResolutionColorSurfelMap::Surfel* src_;
			unsigned int src_idx_;
			spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_dst_;
			MultiResolutionColorSurfelMap::Surfel* dst_;
			unsigned int dst_idx_;

			Eigen::Matrix< double, 6, 1 > df_dx;
			Eigen::Matrix< double, 6, 6 > d2f, JSzJ;
			double error;
			double weight;
			int match;


			// for Levenberg-Marquardt
			// (z - h)^T W (z - h)
			Eigen::Vector3d z, h;
			Eigen::Matrix< double, 3, 6 > dh_dx;
			Eigen::Matrix3d W;


		public:
			EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		};

		typedef std::vector< SurfelAssociation, Eigen::aligned_allocator< SurfelAssociation > > SurfelAssociationList;


		class FeatureAssociation {
		public:
			FeatureAssociation()
			: src_idx_(0), dst_idx_(0), match(0), weight(1.0) {}
			FeatureAssociation( unsigned int src_idx, unsigned int dst_idx )
			: src_idx_(src_idx), dst_idx_(dst_idx), match(1), weight(1.0) {}
			~FeatureAssociation() {}

			unsigned int src_idx_;
			unsigned int dst_idx_;

			double error;
			int match;
			double weight;

			// for direct derivatives of error function
			Eigen::Matrix< double, 6, 1 > df_dx;
			Eigen::Matrix< double, 6, 6 > d2f, JSzJ;

			// AreNo
			Eigen::Vector3d landmark_pos, tmp_landmark_pos;	// estimation for 3D position in source-frame
			Eigen::Matrix<double, 6, 6> Hpp;
			Eigen::Matrix<double, 3, 6> Hpl;
			Eigen::Matrix<double, 3, 3> Hll;
			Eigen::Matrix<double, 6, 1> bp;
			Eigen::Vector3d				 bl;

		public:
			EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		};

		typedef std::vector< FeatureAssociation, Eigen::aligned_allocator< FeatureAssociation > > FeatureAssociationList;


		class NodeLogLikelihood {
		public:
			NodeLogLikelihood()
			: n_(NULL), loglikelihood_(0.0) {}
			NodeLogLikelihood( spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n )
			: n_(n), loglikelihood_(0.0) {}
			~NodeLogLikelihood() {}

			spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_;

			double loglikelihood_;

		};

		typedef std::vector< NodeLogLikelihood > NodeLogLikelihoodList;


		void associateMapsBreadthFirst( SurfelAssociationList& surfelAssociations, MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, algorithm::OcTreeSamplingVectorMap< float, MultiResolutionColorSurfelMap::NodeValue >& targetSamplingMap, Eigen::Matrix4d& transform, double minResolution, double maxResolution );
		void associateMapsBreadthFirstParallel( SurfelAssociationList& surfelAssociations, MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, algorithm::OcTreeSamplingVectorMap< float, MultiResolutionColorSurfelMap::NodeValue >& targetSamplingMap, Eigen::Matrix4d& transform, double minResolution, double maxResolution, double searchDistFactor, double maxSearchDist, bool useFeatures );

		void associateNodeListParallel( SurfelAssociationList& surfelAssociations, MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* >& nodes, int processDepth, Eigen::Matrix4d& transform, double searchDistFactor, double maxSearchDist, bool useFeatures );

		void associatePointFeatures();

		double preparePointFeatureDerivatives( const Eigen::Matrix<double, 6, 1>& x, double qw, double mahaldist );


		std::pair< int, int > calculateNegLogLikelihood( double& logLikelihood, spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* node_src, spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* node_tgt, const Eigen::Matrix4d& transform, double spatial_z_cov_factor, double color_z_cov, double normal_z_cov, bool interpolate );
		spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* calculateNegLogLikelihoodFeatureScoreN( double& logLikelihood, double& featureScore, bool& outOfImage, bool& virtualBorder, bool& occluded, spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* node, const MultiResolutionColorSurfelMap& target, const Eigen::Matrix4d& transform, double spatial_z_cov_factor, double color_z_cov, double normal_z_cov, bool interpolate = false );
		spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* calculateNegLogLikelihoodN( double& logLikelihood, bool& outOfImage, bool& virtualBorder, bool& occluded, spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* node, const MultiResolutionColorSurfelMap& target, const Eigen::Matrix4d& transform, double spatial_z_cov_factor, double color_z_cov, double normal_z_cov, bool interpolate = false );
		bool calculateNegLogLikelihood( double& likelihood, spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* node, const MultiResolutionColorSurfelMap& target, const Eigen::Matrix4d& transform, double spatial_z_cov_factor, double color_z_cov, double normal_z_cov, bool interpolate = false );

		// transform from src to tgt
		double calculateInPlaneLogLikelihood( spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_src, spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_tgt, const Eigen::Matrix4d& transform, double normal_z_cov );


		double matchLogLikelihood( MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, Eigen::Matrix4d& transform );
		double selfMatchLogLikelihood( MultiResolutionColorSurfelMap& target );

		bool estimateTransformationNewton( Eigen::Matrix4d& transform, int coarseToFineIterations, int fineIterations );
		bool estimateTransformationLevenbergMarquardt( Eigen::Matrix4d& transform, int maxIterations );
		bool estimateTransformationLevenbergMarquardtPF( Eigen::Matrix4d& transform, int maxIterations, double featureAssocMahalDist, double minDelta, double& mu, double& nu );

		bool estimateTransformation( MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution, pcl::PointCloud< pcl::PointXYZRGB >::Ptr correspondencesSourcePoints, pcl::PointCloud< pcl::PointXYZRGB >::Ptr correspondencesTargetPoints, int gradientIterations = 100, int coarseToFineIterations = 0, int fineIterations = 5 );


		bool estimatePoseCovariance( Eigen::Matrix< double, 6, 6 >& cov, MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution );
		bool estimatePoseCovarianceLM( Eigen::Matrix< double, 6, 6 >& cov, MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution );


		void setPriorPoseEnabled( bool enabled ) { params_.use_prior_pose_ = enabled; }
		void setPriorPose( bool enabled, const Eigen::Matrix< double, 6, 1 >& prior_pose_mean, const Eigen::Matrix< double, 6, 1 >& prior_pose_variances );

		Params params_;

		MultiResolutionColorSurfelMap* source_;
		MultiResolutionColorSurfelMap* target_;
//		SurfelAssociationList surfelAssociations_;
		FeatureAssociationList featureAssociations_;
		algorithm::OcTreeSamplingVectorMap< float, MultiResolutionColorSurfelMap::NodeValue > targetSamplingMap_;
		float lastWSign_;
		bool interpolate_neighbors_;

		pcl::PointCloud< pcl::PointXYZRGB >::Ptr correspondences_source_points_;
		pcl::PointCloud< pcl::PointXYZRGB >::Ptr correspondences_target_points_;

		// 2.5D --> 3D
		inline Eigen::Vector3d phi(const Eigen::Vector3d& m) const {
			Eigen::Vector3d tmp = m;
			tmp(0) /= tmp(2);
			tmp(1) /= tmp(2);
			tmp(2) = 1 / tmp(2);
			return params_.KInv_ * tmp;
		}

		// 3D --> 2.5D
		inline Eigen::Vector3d phiInv(const Eigen::Vector3d& lm) const {
			Eigen::Vector3d tmp = lm;
			double depth = lm(2);
			tmp = (params_.K_ * tmp).eval();
			tmp(0) /= depth;
			tmp(1) /= depth;
			tmp(2) /= depth * depth;
			return tmp;
		}

		// h( m , x)
		inline Eigen::Vector3d h(const Eigen::Vector3d& m, const Eigen::Matrix3d& rot, const Eigen::Vector3d& trnsl) const {
			return phiInv(rot * phi(m) + trnsl);
		}

	protected:

		bool registrationErrorFunctionWithFirstDerivative( const Eigen::Matrix< double, 6, 1 >& x, double& f, Eigen::Matrix< double, 6, 1 >& df_dx, MultiResolutionColorSurfelRegistration::SurfelAssociationList& surfelAssociations );
		bool registrationErrorFunctionWithFirstAndSecondDerivative( const Eigen::Matrix< double, 6, 1 >& x, bool relativeDerivative, double& f, Eigen::Matrix< double, 6, 1 >& df_dx, Eigen::Matrix< double, 6, 6 >& d2f_dx2, MultiResolutionColorSurfelRegistration::SurfelAssociationList& surfelAssociations );

		bool registrationErrorFunctionLM( const Eigen::Matrix<double, 6, 1>& x, double& f, MultiResolutionColorSurfelRegistration::SurfelAssociationList& surfelAssociations, MultiResolutionColorSurfelRegistration::FeatureAssociationList& featureAssociations, double mahaldist );

		bool registrationSurfelErrorFunctionWithFirstAndSecondDerivativeLM( const Eigen::Matrix< double, 6, 1 >& x, double& f, Eigen::Matrix< double, 6, 1 >& df, Eigen::Matrix< double, 6, 6 >& d2f, MultiResolutionColorSurfelRegistration::SurfelAssociationList& surfelAssociations );


	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	};


//	class MultiFrameMultiResolutionColorSurfelRegistration : public MultiResolutionColorSurfelRegistration {
//	public:
//
//		class FramePair {
//		public:
//
//			FramePair( MultiResolutionColorSurfelMap* source, MultiResolutionColorSurfelMap* target, const Eigen::Matrix4d& transformGuess )
//			: source_( source )
//			, target_( target )
//			, transform_( transformGuess ) {
//			}
//
//			~FramePair() {}
//
//			MultiResolutionColorSurfelMap* source_;
//			MultiResolutionColorSurfelMap* target_;
//			Eigen::Matrix4d transform_;
//
//		public:
//			EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
//		};
//
//
//		class FramePairConstraint {
//		public:
//
//			FramePairConstraint( MultiResolutionColorSurfelMap* target_from, MultiResolutionColorSurfelMap* target_to, const Eigen::Matrix4d& refFrameTransform, const Eigen::Matrix< double, 7, 1 >& poseConstraintMean, const Eigen::Matrix< double, 6, 1 >& poseConstraintVar )
//			: target_from_( target_from )
//			, target_to_( target_to )
//			, ref_frame_transform_( refFrameTransform )
//			, pose_constraint_mean_( poseConstraintMean ) {
//				pose_constraint_invcov_ = Eigen::DiagonalMatrix< double, 6 >( poseConstraintVar ).inverse();
//			}
//
//			~FramePairConstraint() {}
//
//			MultiResolutionColorSurfelMap* target_from_;
//			MultiResolutionColorSurfelMap* target_to_;
//
//			Eigen::Matrix4d ref_frame_transform_;
//
//			Eigen::Matrix< double, 7, 1 > pose_constraint_mean_;
//			Eigen::Matrix< double, 6, 6 > pose_constraint_invcov_;
//
//		public:
//			EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
//		};
//
//
//		MultiFrameMultiResolutionColorSurfelRegistration();
//		~MultiFrameMultiResolutionColorSurfelRegistration() {}
//
//		void setResolutionRange( float startResolution, float stopResolution ) {
//			start_resolution_ = startResolution;
//			stop_resolution_ = stopResolution;
//		}
//
//		void addFramePair( MultiResolutionColorSurfelMap* source, MultiResolutionColorSurfelMap* target, const Eigen::Matrix4d& transformGuess );
//
//		void addTargetPoseConstraint( MultiResolutionColorSurfelMap* target_from, MultiResolutionColorSurfelMap* target_to, const Eigen::Matrix4d& refFrameTransform, const Eigen::Matrix< double, 7, 1 >& prior_pose_mean, const Eigen::Matrix< double, 6, 1 >& prior_pose_variances );
//
//
//		bool estimateTransformationLevenbergMarquardt( unsigned int maxIterations );
//		bool estimateTransformationNewton( unsigned int maxIterations );
//
//		bool estimateTransformation( unsigned int levmarIterations = 100, unsigned int newtonIterations = 5 );
//
//
//
//	//protected:
//	public:
//
//		Eigen::Matrix< double, 6, 1 > dpose_dT_times_dT_ddelta( const Eigen::Matrix4d& dT_ddelta );
//		void poseConstraintError( Eigen::Matrix< double, 6, 1 >& poseDiff, const Eigen::Matrix4d& refFrameTransform, const Eigen::Matrix< double, 7, 1 >& poseConstraint, const Eigen::Matrix4d& poseEstimateFrom, const Eigen::Matrix4d& poseEstimateTo );
//		void poseConstraintErrorWithFirstDerivative( Eigen::Matrix< double, 6, 1 >& poseDiff, Eigen::Matrix< double, 6, 6 >& J_from, Eigen::Matrix< double, 6, 6 >& J_to, const Eigen::Matrix4d& refFrameTransform, const Eigen::Matrix< double, 7, 1 >& poseConstraint, const Eigen::Matrix4d& poseEstimateFrom, const Eigen::Matrix4d& poseEstimateTo );
//
//
//		std::vector< FramePair, Eigen::aligned_allocator< FramePair > > registration_pairs_;
//		std::vector< FramePairConstraint, Eigen::aligned_allocator< FramePairConstraint > > target_pose_constraints_;
//		std::vector< std::pair< int, int > > target_pose_constraint_ids_;
//
//		std::vector< algorithm::OcTreeSamplingVectorMap<float, MultiResolutionColorSurfelMap::NodeValue>, Eigen::aligned_allocator< algorithm::OcTreeSamplingVectorMap<float, MultiResolutionColorSurfelMap::NodeValue> > > target_sampling_maps_;
//
//		Eigen::Matrix4d dT_tx_, dT_ty_, dT_tz_, dT_qx_, dT_qy_, dT_qz_;
//
//		float start_resolution_, stop_resolution_;
//
//	public:
//		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
//	};

};


#endif /* MULTIRESOLUTION_SURFEL_REGISTRATION_H_ */


