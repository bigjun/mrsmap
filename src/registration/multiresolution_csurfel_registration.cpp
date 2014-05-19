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

#include "mrsmap/registration/multiresolution_csurfel_registration.h"

#include <mrsmap/utilities/utilities.h>

#include <g2o/types/slam3d/dquat2mat.h>

#include <deque>

#include <fstream>

#include <tbb/tbb.h>

using namespace mrsmap;

typedef MultiResolutionColorSurfelRegistration MRCSReg;
typedef MRCSReg::SurfelAssociationList MRCSRSAL;
typedef MRCSReg::FeatureAssociationList MRCSRFAL;
typedef MultiResolutionColorSurfelMap MRCSMap;


// TODO: falcopy vermeiden, prüfen, wieviele iterationen in coarse verbracht werden..


MultiResolutionColorSurfelRegistration::Params::Params() {
	init();
}

void MultiResolutionColorSurfelRegistration::Params::init() {

	// defaults
	registerSurfels_ = true;
	registerFeatures_ = false;


	use_prior_pose_ = false;
	prior_pose_mean_ = Eigen::Matrix< double, 6, 1 >::Zero();
	prior_pose_invcov_ = Eigen::Matrix< double, 6, 6 >::Identity();

	add_smooth_pos_covariance_ = true;
	smooth_surface_cov_factor_ = 0.001f;

	surfel_match_angle_threshold_ = 0.5;
	registration_min_num_surfels_ = 0;
	max_feature_dist2_ = 0.1;
	use_features_ = true;

	match_likelihood_use_color_ = true;
	luminance_damp_diff_ = 0.5;
	color_damp_diff_ = 0.1;

	registration_use_color_ = true;
	luminance_reg_threshold_ = 0.5;
	color_reg_threshold_ = 0.1;

	occlusion_z_similarity_factor_ = 0.02f;
	image_border_range_ = 40;

	parallel_ = true;


	startResolution_ = 0.0125f;
	stopResolution_ = 0.2f;


	pointFeatureMatchingNumNeighbors_= 3;
	pointFeatureMatchingThreshold_ = 40;
	pointFeatureMatchingCoarseImagePosMahalDist_ = 1000.0;
	pointFeatureMatchingFineImagePosMahalDist_ = 48.0;
	pointFeatureWeight_ = 0.05; // weighting relative to surfels
	// matchings beyond this threshold are considered outliers
	debugFeatures_ = false;
	calibration_f_ = 525.f;
	calibration_c1_ = 319.5f;
	calibration_c2_ = 239.5f;
	K_.setIdentity();
	K_(0, 0) = K_(1, 1) = calibration_f_;
	K_(0, 2) = calibration_c1_;
	K_(1, 2) = calibration_c2_;
	KInv_ = K_.inverse();

}

std::string MultiResolutionColorSurfelRegistration::Params::toString() {

	std::stringstream retVal;

	retVal << "use_prior_pose: " << (use_prior_pose_ ? 1 : 0) << std::endl;
	retVal << "prior_pose_mean: " << prior_pose_mean_.transpose() << std::endl;
	retVal << "prior_pose_invcov: " << prior_pose_invcov_ << std::endl;

	return retVal.str();

}



MultiResolutionColorSurfelRegistration::MultiResolutionColorSurfelRegistration() {

}


MultiResolutionColorSurfelRegistration::MultiResolutionColorSurfelRegistration( const Params& params ) {

	params_ = params;

}


void MultiResolutionColorSurfelRegistration::setPriorPose( bool enabled, const Eigen::Matrix< double, 6, 1 >& prior_pose_mean, const Eigen::Matrix< double, 6, 1 >& prior_pose_variances ) {

	params_.use_prior_pose_ = enabled;
	params_.prior_pose_mean_ = prior_pose_mean;
	params_.prior_pose_invcov_ = Eigen::DiagonalMatrix< double, 6 >( prior_pose_variances ).inverse();

}


bool pointOccluded( const Eigen::Vector4f& p, const MultiResolutionColorSurfelMap& target, double z_similarity_factor ) {

	if( isnan( p(0) ) )
		return false;

	int px = 525.0 * p(0) / p(2) + 319.5;
	int py = 525.0 * p(1) / p(2) + 239.5;


	if( px < 0 || px >= 640 || py < 0 || py >= 480 ) {
		return false;
	}

	if( !target.imageAllocator_->node_set_.empty() ) {

		unsigned int idx = py * 640 + px;
		const spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n = target.imageAllocator_->node_image_[idx];
		if( n ) {
			double z_dist = std::max( 0.f, p(2) - n->getCenterPosition()(2) );
			if( z_dist > fabsf( z_similarity_factor * p(2) ) )
				return true;
		}

	}
	else {

		std::cout << "WARNING: mrsmap not created with node image! occlusion check disabled.\n";

	}


	return false;

}


spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* MultiResolutionColorSurfelRegistration::calculateNegLogLikelihoodFeatureScoreN( double& logLikelihood, double& featureScore, bool& outOfImage, bool& virtualBorder, bool& occluded, spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* node, const MultiResolutionColorSurfelMap& target, const Eigen::Matrix4d& transform, double spatial_z_cov_factor, double color_z_cov, double normal_z_cov, bool interpolate ) {

	// for each surfel in node with applyUpdate set and sufficient points, transform to target using transform,
	// then measure negative log likelihood

//	const double spatial_z_cov_factor = 0.04;
//	const double color_z_cov = 0.0001;
//	const double normalStd = 0.25*M_PI;

	featureScore = std::numeric_limits<double>::max();
	logLikelihood = std::numeric_limits<double>::max();

	Eigen::Matrix3d rotation = transform.block<3,3>(0,0);

	// determine corresponding node in target..
	Eigen::Vector4f npos = node->getPosition();
	npos(3) = 1.0;
	Eigen::Vector4f npos_match_src = transform.cast<float>() * npos;

	if( !pointInImage( npos_match_src, params_.image_border_range_ ) )
		outOfImage = true;

	// also check if point is occluded (project into image and compare z coordinate at some threshold)
	occluded = pointOccluded( npos_match_src, target, params_.occlusion_z_similarity_factor_ );
//	if( occluded )
//		return NULL;

	std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* > neighbors;
	neighbors.reserve(50);
	float searchRange = 2.f;
	Eigen::Vector4f minPosition = npos_match_src - Eigen::Vector4f( searchRange*node->resolution(), searchRange*node->resolution(), searchRange*node->resolution(), 0.f );
	Eigen::Vector4f maxPosition = npos_match_src + Eigen::Vector4f( searchRange*node->resolution(), searchRange*node->resolution(), searchRange*node->resolution(), 0.f );

	target.octree_->getAllNodesInVolumeOnDepth( neighbors, minPosition, maxPosition, node->depth_, true );

	if( neighbors.size() == 0 ) {
		return NULL;
	}

	spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_matched = NULL;
	MultiResolutionColorSurfelMap::Surfel* srcSurfel = NULL;
	MultiResolutionColorSurfelMap::Surfel* matchedSurfel = NULL;
	int matchedSurfelIdx = -1;
	double bestDist = std::numeric_limits<double>::max();

	// get closest node in neighbor list
	for( unsigned int i = 0; i < MAX_NUM_SURFELS; i++ ) {

		MultiResolutionColorSurfelMap::Surfel& surfel = node->value_.surfels_[i];

		// border points are returned but must be handled later!
		if( surfel.num_points_ < MIN_SURFEL_POINTS ) {
			continue;
		}
//		if( surfel.num_points_ < MIN_SURFEL_POINTS || !surfel.applyUpdate_ ) {
//			continue;
//		}

		Eigen::Vector4d pos;
		pos.block<3,1>(0,0) = surfel.mean_.block<3,1>(0,0);
		pos(3,0) = 1.f;

		Eigen::Vector4d pos_match_src = transform * pos;
		Eigen::Vector3d dir_match_src = rotation * surfel.initial_view_dir_;

		for( std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* >::iterator it = neighbors.begin(); it != neighbors.end(); it++ ) {

			if( (*it)->value_.border_ != node->value_.border_ )
				continue;

			MultiResolutionColorSurfelMap::Surfel* bestMatchSurfel = NULL;
			int bestMatchSurfelIdx = -1;
			double bestMatchDist = -1.f;
			for( unsigned int k = 0; k < MAX_NUM_SURFELS; k++ ) {

				const MultiResolutionColorSurfelMap::Surfel& srcSurfel2 = (*it)->value_.surfels_[k];

				if( srcSurfel2.num_points_ < MIN_SURFEL_POINTS ) {
					continue;
				}

				const double dist = dir_match_src.dot( srcSurfel2.initial_view_dir_ );
				if( dist >= params_.surfel_match_angle_threshold_ && dist >= bestMatchDist ) {
					bestMatchSurfel = &((*it)->value_.surfels_[k]);
					bestMatchDist = dist;
					bestMatchSurfelIdx = k;
				}
			}

			if( bestMatchSurfel ) {
				// use distance between means
				double dist = (pos_match_src.block<3,1>(0,0) - bestMatchSurfel->mean_.block<3,1>(0,0)).norm();
				if( dist < bestDist ) {
					bestDist = dist;
					srcSurfel = &surfel;
					n_matched = *it;
					matchedSurfel = bestMatchSurfel;
					matchedSurfelIdx = bestMatchSurfelIdx;
				}
			}
		}

	}

	// border points are returned but must be handled later!
//	if( !n_matched || !matchedSurfel->applyUpdate_ ) {
	if( !n_matched ) {
		return NULL;
	}

	if( !srcSurfel->applyUpdate_ || !matchedSurfel->applyUpdate_ )
		virtualBorder = true;

//	if( !matchedSurfel->applyUpdate_ )
//		virtualBorder = true;
//	if( !srcSurfel->applyUpdate_ )
//		return NULL;

	featureScore = 0;//srcSurfel->agglomerated_shape_texture_features_.distance( matchedSurfel->agglomerated_shape_texture_features_ );

	Eigen::Vector4d pos;
	pos.block<3,1>(0,0) = srcSurfel->mean_.block<3,1>(0,0);
	pos(3,0) = 1.f;

	Eigen::Vector4d pos_match_src = transform * pos;

	double l = 0;


	if( params_.match_likelihood_use_color_ ) {

//		Eigen::Matrix< double, 6, 6 > rotation6 = Eigen::Matrix< double, 6, 6 >::Identity();
//		rotation6.block<3,3>(0,0) = rotation;

		Eigen::Matrix< double, 6, 6 > cov1;
		Eigen::Matrix< double, 6, 1 > dstMean;

		bool in_interpolation_range = true;

		if( interpolate ) {

			// use trilinear interpolation to handle discretization effects
			// => associate with neighbors and weight correspondences
			// only makes sense when match is within resolution distance to the node center

			// associate with neighbors for which distance to the node center is smaller than resolution

			dstMean.setZero();
			cov1.setZero();

			double sumWeight = 0.f;
			double sumWeight2 = 0.f;

			const float resolution = n_matched->resolution();

			for( int s = 0; s < 27; s++ ) {

				spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_dst_n = n_matched->neighbors_[s];

				if(!n_dst_n)
					continue;

				MultiResolutionColorSurfelMap::Surfel* dst_n = &n_dst_n->value_.surfels_[matchedSurfelIdx];
				if( dst_n->num_points_ < MIN_SURFEL_POINTS )
					continue;

				Eigen::Vector3d centerDiff_n = n_dst_n->getCenterPosition().block<3,1>(0,0).cast<double>() - pos_match_src.block<3,1>(0,0);
				const double dx = resolution - fabsf(centerDiff_n(0));
				const double dy = resolution - fabsf(centerDiff_n(1));
				const double dz = resolution - fabsf(centerDiff_n(2));

				if( dx > 0 && dy > 0 && dz > 0 ) {

					const double weight = dx*dy*dz;

					dstMean += weight * dst_n->mean_;
					cov1 += weight*weight * (dst_n->cov_);

					sumWeight += weight;
					sumWeight2 += weight*weight;

				}


			}

			// numerically stable?
			if( sumWeight > resolution* 1e-6 ) {
				dstMean /= sumWeight;
				cov1 /= sumWeight2;

			}
			else
				in_interpolation_range = false;

		}

		if( !interpolate || !in_interpolation_range ) {

			dstMean = matchedSurfel->mean_;
			cov1 = matchedSurfel->cov_;

		}


		// has only marginal (positive!) effect on visual odometry result
		// makes tracking more robust (when only few surfels available)
//		cov1 *= INTERPOLATION_COV_FACTOR;
//		const Eigen::Matrix< double, 6, 6 > cov2 = INTERPOLATION_COV_FACTOR * srcSurfel->cov_;

		const Eigen::Matrix< double, 6, 6 > cov2 = srcSurfel->cov_;

		Eigen::Matrix< double, 6, 1 > diff;
		diff.block<3,1>(0,0) = dstMean.block<3,1>(0,0) - pos_match_src.block<3,1>(0,0);
		diff.block<3,1>(3,0) = dstMean.block<3,1>(3,0) - srcSurfel->mean_.block<3,1>(3,0);
		if( fabs(diff(3)) < params_.luminance_damp_diff_ )
			diff(3) = 0;
		if( fabs(diff(4)) < params_.color_damp_diff_ )
			diff(4) = 0;
		if( fabs(diff(5)) < params_.color_damp_diff_ )
			diff(5) = 0;

		if( diff(3) < 0 )
			diff(3) += params_.luminance_damp_diff_;
		if( diff(4) < 0 )
			diff(4) += params_.color_damp_diff_;
		if( diff(5) < 0 )
			diff(5) += params_.color_damp_diff_;

		if( diff(3) > 0 )
			diff(3) -= params_.luminance_damp_diff_;
		if( diff(4) > 0 )
			diff(4) -= params_.color_damp_diff_;
		if( diff(5) > 0 )
			diff(5) -= params_.color_damp_diff_;


		// has only marginal (positive!) effect on visual odometry result
		// makes tracking more robust (when only few surfels available)
		const Eigen::Matrix3d cov1_ss = cov1.block<3,3>(0,0);
		const Eigen::Matrix3d cov2_ss = cov2.block<3,3>(0,0);

		const Eigen::Matrix3d Rcov2_ss = rotation * cov2_ss;

		const Eigen::Matrix3d cov_ss = cov1_ss + Rcov2_ss * rotation.transpose() + spatial_z_cov_factor*node->resolution()*node->resolution() * Eigen::Matrix3d::Identity();
//		const Eigen::Matrix3d cov_ss = cov1_ss + Rcov2_ss * rotation.transpose() + spatial_z_cov_factor*0.01 * Eigen::Matrix3d::Identity();
//		const Eigen::Matrix3d cov_ss = node->resolution()*node->resolution() * Eigen::Matrix3d::Identity();
		const Eigen::Matrix3d invcov_ss = cov_ss.inverse();

		const Eigen::Vector3d invcov_ss_diff_s = invcov_ss * diff.block<3,1>(0,0);

//		l = log( cov_ss.determinant() ) + diff.block<3,1>(0,0).dot(invcov_ss_diff_s);
		l = diff.block<3,1>(0,0).dot(invcov_ss_diff_s);
//		l = std::max( 0.0, l - 9.0 );
//		if( l < 9 )
//			l = 0;
		if( l > 48.0 )
			l = 48.0;

//		std::cout << "s:\n";
//		std::cout << diff.block<3,1>(0,0) << "\n";
//		std::cout << cov_ss.block<3,3>(0,0) << "\n";
//		std::cout << log( cov_ss.determinant() ) << "\n";
//		std::cout << l << "\n";

		const Eigen::Matrix3d cov_cc = cov1.block<3,3>(3,3) + cov2.block<3,3>(3,3) + color_z_cov * Eigen::Matrix3d::Identity();
//		l += log( cov_cc.determinant() ) + diff.block<3,1>(3,0).dot( cov_cc.inverse() * diff.block<3,1>(3,0) );
		double color_loglikelihood = diff.block<3,1>(3,0).dot( (cov_cc.inverse() * diff.block<3,1>(3,0)).eval() );
		if( color_loglikelihood > 48.0 )
			color_loglikelihood = 48.0;
		l += color_loglikelihood;


	}
	else {

		Eigen::Matrix3d cov1_ss;
		Eigen::Vector3d dstMean;

		bool in_interpolation_range = true;

		if( interpolate ) {

			// use trilinear interpolation to handle discretization effects
			// => associate with neighbors and weight correspondences
			// only makes sense when match is within resolution distance to the node center
			const float resolution = node->resolution();

			// associate with neighbors for which distance to the node center is smaller than resolution

			dstMean.setZero();
			cov1_ss.setZero();

			double sumWeight = 0.f;
			double sumWeight2 = 0.f;

			for( int s = 0; s < 27; s++ ) {

				spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_dst_n = n_matched->neighbors_[s];

				if(!n_dst_n)
					continue;

				MultiResolutionColorSurfelMap::Surfel* dst_n = &n_dst_n->value_.surfels_[matchedSurfelIdx];
				if( dst_n->num_points_ < MIN_SURFEL_POINTS )
					continue;

				Eigen::Vector3d centerDiff_n = n_dst_n->getCenterPosition().block<3,1>(0,0).cast<double>() - pos_match_src.block<3,1>(0,0);
				const double dx = resolution - fabsf(centerDiff_n(0));
				const double dy = resolution - fabsf(centerDiff_n(1));
				const double dz = resolution - fabsf(centerDiff_n(2));

				if( dx > 0 && dy > 0 && dz > 0 ) {

					const double weight = dx*dy*dz;

					dstMean += weight * dst_n->mean_.block<3,1>(0,0);
					cov1_ss += weight*weight * (dst_n->cov_.block<3,3>(0,0));

					sumWeight += weight;
					sumWeight2 += weight*weight;

				}


			}

			// numerically stable?
			if( sumWeight > resolution* 1e-6 ) {
				dstMean /= sumWeight;
				cov1_ss /= sumWeight2;

			}
			else
				in_interpolation_range = false;

		}

		if( !interpolate || !in_interpolation_range ) {

			dstMean = matchedSurfel->mean_.block<3,1>(0,0);
			cov1_ss = matchedSurfel->cov_.block<3,3>(0,0);

		}


		// has only marginal (positive!) effect on visual odometry result
		// makes tracking more robust (when only few surfels available)
		cov1_ss *= INTERPOLATION_COV_FACTOR;
		const Eigen::Matrix3d cov2_ss = INTERPOLATION_COV_FACTOR * srcSurfel->cov_.block<3,3>(0,0);

		const Eigen::Vector3d diff_s = dstMean - pos_match_src.block<3,1>(0,0);

		const Eigen::Matrix3d Rcov2_ss = rotation * cov2_ss;

		const Eigen::Matrix3d cov_ss = cov1_ss + Rcov2_ss * rotation.transpose() + spatial_z_cov_factor*node->resolution()*node->resolution() * Eigen::Matrix3d::Identity();
		const Eigen::Matrix3d invcov_ss = cov_ss.inverse();

		const Eigen::Vector3d invcov_ss_diff_s = invcov_ss * diff_s;

//		l = log( cov_ss.determinant() ) + diff_s.dot(invcov_ss_diff_s);
		l = diff_s.dot(invcov_ss_diff_s);
		if( l > 48.0 )
			l = 48.0;

	}


	// also consider normal orientation in the likelihood
//	// TODO curvature-dependency should be made nicer
//	if( srcSurfel->surface_curvature_ < 0.05 ) {
		Eigen::Vector4d normal_src;
		normal_src.block<3,1>(0,0) = srcSurfel->normal_;
		normal_src(3,0) = 0.0;
		normal_src = (transform * normal_src).eval();

		double normalError = acos( normal_src.block<3,1>(0,0).dot( matchedSurfel->normal_ ) );
		double normalExponent = std::min( 4.0, normalError * normalError / normal_z_cov );
	//	double normalLogLikelihood = log( 2.0 * M_PI * normalStd ) + normalExponent;

		l += normalExponent;
	//	l += normalLogLikelihood;

	//	std::cout << "n:\n";
	//	std::cout << normalError << "\n";
	//	std::cout << normalExponent << "\n\n";
//	}

	logLikelihood = std::min( l, logLikelihood );

	return n_matched;

}


spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* MultiResolutionColorSurfelRegistration::calculateNegLogLikelihoodN( double& logLikelihood, bool& outOfImage, bool& virtualBorder, bool& occluded, spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* node, const MultiResolutionColorSurfelMap& target, const Eigen::Matrix4d& transform, double spatial_z_cov_factor, double color_z_cov, double normal_z_cov, bool interpolate ) {

	double featureScore = 0.0;

	return calculateNegLogLikelihoodFeatureScoreN( logLikelihood, featureScore, outOfImage, virtualBorder, occluded, node, target, transform, spatial_z_cov_factor, color_z_cov, normal_z_cov, interpolate );

}


bool MultiResolutionColorSurfelRegistration::calculateNegLogLikelihood( double& logLikelihood, spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* node, const MultiResolutionColorSurfelMap& target, const Eigen::Matrix4d& transform, double spatial_z_cov_factor, double color_z_cov, double normal_z_cov, bool interpolate ) {

	bool outOfImage = false;
	bool virtualBorder = false;
	bool occluded = false;
	if( calculateNegLogLikelihoodN( logLikelihood, outOfImage, virtualBorder, occluded, node, target, transform, spatial_z_cov_factor, color_z_cov, normal_z_cov, interpolate ) != NULL )
		return true;
	else
		return false;

}


// transform from src to tgt
double MultiResolutionColorSurfelRegistration::calculateInPlaneLogLikelihood( spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_src, spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_tgt, const Eigen::Matrix4d& transform, double normal_z_cov ) {

	double bestLogLikelihood = 18.0;
	for( unsigned int i = 0; i < MAX_NUM_SURFELS; i++ ) {

		ColorSurfel& s_src = n_src->value_.surfels_[i];
		ColorSurfel& s_tgt = n_tgt->value_.surfels_[i];

		if( s_src.num_points_ < MIN_SURFEL_POINTS ) {
			continue;
		}

		if( s_tgt.num_points_ < MIN_SURFEL_POINTS ) {
			continue;
		}

		// measure variance along normal direction of reference surfel
		const Eigen::Vector3d mean_src = s_src.mean_.block<3,1>(0,0);
		const Eigen::Vector3d mean_tgt = s_tgt.mean_.block<3,1>(0,0);
		const Eigen::Vector4d mean_src4( mean_src(0), mean_src(1), mean_src(2), 1.0 );
		const Eigen::Vector3d mean_src_transformed = (transform * mean_src4).block<3,1>(0,0);

		const Eigen::Matrix3d rot_src = transform.block<3,3>(0,0);
		const Eigen::Matrix3d cov_src_transformed = rot_src * (s_src.cov_.block<3,3>(0,0)) * rot_src.transpose();
		const Eigen::Matrix3d cov_tgt = s_tgt.cov_.block<3,3>(0,0);

		const Eigen::Vector3d n_tgt = s_tgt.normal_;

		double var_n_src = n_tgt.transpose() * cov_src_transformed * n_tgt;
		double var_n_tgt = n_tgt.transpose() * cov_tgt * n_tgt;
		double var_n = var_n_src + var_n_tgt;

		double diff_n = n_tgt.dot( mean_tgt - mean_src_transformed );

		double logLikelihood = diff_n*diff_n / var_n;


		// also consider normal orientation for the likelihood
		Eigen::Vector4d normal_src;
		normal_src.block<3,1>(0,0) = s_src.normal_;
		normal_src(3,0) = 0.0;
		normal_src = (transform * normal_src).eval();

		double normalError = acos( normal_src.block<3,1>(0,0).dot( s_tgt.normal_ ) );
		double normalExponent = std::min( 9.0, normalError * normalError / normal_z_cov );

		logLikelihood += normalExponent;


		bestLogLikelihood = std::min( bestLogLikelihood, logLikelihood );

	}

	return bestLogLikelihood;

}


void MultiResolutionColorSurfelRegistration::associateMapsBreadthFirstParallel( MultiResolutionColorSurfelRegistration::SurfelAssociationList& surfelAssociations, MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, algorithm::OcTreeSamplingVectorMap< float, MultiResolutionColorSurfelMap::NodeValue >& targetSamplingMap, Eigen::Matrix4d& transform, double minResolution, double maxResolution, double searchDistFactor, double maxSearchDist, bool useFeatures ) {


	target.distributeAssociatedFlag();

	int maxDepth = std::min( source.octree_->max_depth_, target.octree_->max_depth_ );

	// start at coarsest resolution
	// if all children associated, skip the node,
	// otherwise
	// - if already associated from previous iteration, search in local neighborhood
	// - if not associated in previous iteration, but parent has been associated, choose among children of parent's match
	// - otherwise, search in local volume for matches

	int countNodes = 0;
	for( int d = maxDepth; d >= 0; d-- ) {

		const float processResolution = source.octree_->volumeSizeForDepth( d );

		if( processResolution < minResolution || processResolution > maxResolution ) {
			continue;
		}

		countNodes += targetSamplingMap[d].size();

	}
	surfelAssociations.reserve( countNodes );

	for( int d = maxDepth; d >= 0; d-- ) {

		const float processResolution = source.octree_->volumeSizeForDepth( d );

		if( processResolution < minResolution || processResolution > maxResolution ) {
			continue;
		}

		associateNodeListParallel( surfelAssociations, source, target, targetSamplingMap[d], d, transform, searchDistFactor, maxSearchDist, useFeatures );

	}


}


class AssociateFunctor {
public:
	AssociateFunctor( tbb::concurrent_vector< MultiResolutionColorSurfelRegistration::SurfelAssociation >* associations, const MultiResolutionColorSurfelRegistration::Params& params, MultiResolutionColorSurfelMap* source, MultiResolutionColorSurfelMap* target, std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* >* nodes, const Eigen::Matrix4d& transform, int processDepth, double searchDistFactor, double maxSearchDist, bool useFeatures ) {
		associations_ = associations;
		params_ = params;
		source_ = source;
		target_ = target;
		nodes_ = nodes;
		transform_ = transform;
		transformf_ = transform.cast<float>();
		rotation_ = transform.block<3,3>(0,0);

		process_depth_ = processDepth;
		process_resolution_ = source_->octree_->volumeSizeForDepth( processDepth );
		search_dist_ = std::min( searchDistFactor*process_resolution_, maxSearchDist );
		search_dist2_ = search_dist_*search_dist_;
		search_dist_vec_ = Eigen::Vector4f( search_dist_, search_dist_, search_dist_, 0.f );

		use_features_ = useFeatures;

		num_vol_queries_ = 0;
		num_finds_ = 0;
		num_neighbors_ = 0;


	}

	~AssociateFunctor() {}

	void operator()( const tbb::blocked_range<size_t>& r ) const {
		for( size_t i=r.begin(); i!=r.end(); ++i )
			(*this)((*nodes_)[i]);
	}


	void operator()( spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >*& node ) const {

		spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n = node;

		if( n->value_.associated_ == -1 )
			return;

		// all children associated?
		int numAssociatedChildren = 0;
		int numChildren = 0;
		if( n->type_ != spatialaggregate::OCTREE_MAX_DEPTH_BRANCHING_NODE ) {
			for( unsigned int i = 0; i < 8; i++ ) {
				if( n->children_[i] ) {
					numChildren++;
					if( n->children_[i]->value_.associated_ == 1 )
						numAssociatedChildren++;
				}
			}

//			if( numAssociatedChildren > 0 )
//				n->value_.associated_ = 1;
//
//			if( numChildren > 0 && numChildren == numAssociatedChildren )
//				return;

			if( numChildren > 0 && numAssociatedChildren > 0 ) {
				n->value_.associated_ = 1;
				return;
			}
		}

//		if( !n->value_.associated_ )
//			return;


		// check if surfels exist and can be associated by view direction
		// use only one best association per node
		float bestAssocDist = std::numeric_limits<float>::max();
		float bestAssocFeatureDist = std::numeric_limits<float>::max();
		MultiResolutionColorSurfelRegistration::SurfelAssociation bestAssoc;

		bool hasSurfel = false;

		// TODO: collect features for view directions (surfels)
		// once a representative node is chosen, search for feature correspondences by sweeping up the tree up to a maximum search distance.
		// check compatibility using inverse depth parametrization

		// check if a surfels exist
		for( unsigned int i = 0; i < MAX_NUM_SURFELS; i++ ) {

			// if image border points fall into this node, we must check the children_
			if( !n->value_.surfels_[i].applyUpdate_ ) {
				continue;
			}

			if( n->value_.surfels_[i].num_points_ < MIN_SURFEL_POINTS ) {
				continue;
			}

			hasSurfel = true;
		}

		if( hasSurfel ) {

			spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_src_last = NULL;
			std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* > neighbors;

			// association of this node exists from a previous iteration?
			char surfelSrcIdx = -1;
			char surfelDstIdx = -1;
			if( n->value_.association_ ) {
				n_src_last = n->value_.association_;
				surfelSrcIdx = n->value_.assocSurfelIdx_;
				surfelDstIdx = n->value_.assocSurfelDstIdx_;
				n_src_last->getNeighbors( neighbors );
			}

			// does association of parent exist from a previous iteration?
			if( !n_src_last ) {

				if( false && n->parent_ && n->parent_->value_.association_ ) {

					n_src_last = n->parent_->value_.association_;
					surfelSrcIdx = n->parent_->value_.assocSurfelIdx_;
					surfelDstIdx = n->parent_->value_.assocSurfelDstIdx_;

					Eigen::Vector4f npos = n->getCenterPosition();
					npos(3) = 1.f;
					Eigen::Vector4f npos_match_src = transformf_ * npos;

					n_src_last = n_src_last->findRepresentative( npos_match_src, process_depth_ );

					if( n_src_last )
						n_src_last->getNeighbors( neighbors );

				}
				else  {

					neighbors.reserve(50);

					Eigen::Vector4f npos = n->getCenterPosition();
					npos(3) = 1.f;
					Eigen::Vector4f npos_match_src = transformf_ * npos;

					// if direct look-up fails, perform a region query
					// in case there is nothing within the volume, the query will exit early

					Eigen::Vector4f minPosition = npos_match_src - search_dist_vec_;
					Eigen::Vector4f maxPosition = npos_match_src + search_dist_vec_;

					source_->octree_->getAllNodesInVolumeOnDepth( neighbors, minPosition, maxPosition, process_depth_, false );

				}

			}

			if( neighbors.size() == 0 ) {

				n->value_.association_ = NULL;
				n->value_.associated_ = 0;

				return;
			}


			if( surfelSrcIdx >= 0 && surfelDstIdx >= 0 ) {

				const MultiResolutionColorSurfelMap::Surfel& surfel = n->value_.surfels_[surfelSrcIdx];

				if( surfel.num_points_ >= MIN_SURFEL_POINTS ) {

					Eigen::Vector4d pos;
					pos.block<3,1>(0,0) = surfel.mean_.block<3,1>(0,0);
					pos(3,0) = 1.f;

					Eigen::Vector4d pos_match_src = transform_ * pos;
					Eigen::Vector3d dir_match_src = rotation_ * surfel.initial_view_dir_;

					// iterate through neighbors of the directly associated node to eventually find a better match
					for( std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* >::iterator nit = neighbors.begin(); nit != neighbors.end(); ++nit ) {

						spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_src = *nit;

						if( !n_src )
							continue;

						if( n->value_.border_ != n_src->value_.border_ )
							continue;


						// find matching surfel for the view direction, but allow to use a slightly worse fit,
						// when it is the only one with sufficient points for matching
						MultiResolutionColorSurfelMap::Surfel& dstSurfel = n_src->value_.surfels_[surfelDstIdx];

						if( dstSurfel.num_points_ < MIN_SURFEL_POINTS )
							continue;

						const double dist = dir_match_src.dot( dstSurfel.initial_view_dir_ );

						MultiResolutionColorSurfelMap::Surfel* bestMatchSurfel = NULL;
						int bestMatchSurfelIdx = -1;
						double bestMatchDist = -1.f;

						if( dist >= params_.surfel_match_angle_threshold_ ) {
							bestMatchSurfel = &dstSurfel;
							bestMatchDist = dist;
							bestMatchSurfelIdx = surfelDstIdx;
						}

						if( !bestMatchSurfel ) {
							continue;
						}

						// calculate error metric for matching surfels
						double dist_pos2 = (bestMatchSurfel->mean_.block<3,1>(0,0) - pos_match_src.block<3,1>(0,0)).squaredNorm();

						if( dist_pos2 > search_dist2_ )
							continue;


						if( params_.registration_use_color_ ) {
							Eigen::Matrix< double, 3, 1 > diff;
							diff = bestMatchSurfel->mean_.block<3,1>(3,0) - surfel.mean_.block<3,1>(3,0);
							if( fabs(diff(0)) > params_.luminance_reg_threshold_ )
								continue;
							if( fabs(diff(1)) > params_.color_reg_threshold_ )
								continue;
							if( fabs(diff(2)) > params_.color_reg_threshold_ )
								continue;
						}


						// check local descriptor in any case
						float featureDist = 0.0;
						if( use_features_ )
							featureDist = surfel.agglomerated_shape_texture_features_.distance( bestMatchSurfel->agglomerated_shape_texture_features_ );
						if( use_features_ && featureDist > params_.max_feature_dist2_ )
							continue;




						float assocDist = sqrtf(dist_pos2);// + process_resolution_*process_resolution_*(bestMatchSurfel->mean_.block<3,1>(3,0) - surfel.mean_.block<3,1>(3,0)).squaredNorm());

						if( use_features_ )
							assocDist *= featureDist;

						if( assocDist < bestAssocDist ) {
							bestAssocDist = assocDist;
							bestAssocFeatureDist = featureDist;
							n->value_.surfels_[surfelSrcIdx].assocDist_ = assocDist;

//								bestAssoc = MultiResolutionColorSurfelRegistration::SurfelAssociation( n, &n->value_.surfels_[surfelSrcIdx], surfelSrcIdx, n_src, bestMatchSurfel, bestMatchSurfelIdx );
							bestAssoc.n_src_ = n;
							bestAssoc.src_ = &n->value_.surfels_[surfelSrcIdx];
							bestAssoc.src_idx_ = surfelSrcIdx;
							bestAssoc.n_dst_ = n_src;
							bestAssoc.dst_ = bestMatchSurfel;
							bestAssoc.dst_idx_ = bestMatchSurfelIdx;
							bestAssoc.match = 1;

							if( use_features_ )
								bestAssoc.weight = params_.max_feature_dist2_ - featureDist;
							else
								bestAssoc.weight = (1+numChildren) * 1.f;

						}

					}

				}

			}
			else {


				for( unsigned int i = 0; i < MAX_NUM_SURFELS; i++ ) {

					const MultiResolutionColorSurfelMap::Surfel& surfel = n->value_.surfels_[i];

					if( surfel.num_points_ < MIN_SURFEL_POINTS ) {
						continue;
					}

					// transform surfel mean with current transform and find corresponding node in source for current resolution
					// find corresponding surfel in node via the transformed view direction of the surfel

					Eigen::Vector4d pos;
					pos.block<3,1>(0,0) = surfel.mean_.block<3,1>(0,0);
					pos(3,0) = 1.f;

					Eigen::Vector4d pos_match_src = transform_ * pos;
					Eigen::Vector3d dir_match_src = rotation_ * surfel.initial_view_dir_;

					// iterate through neighbors of the directly associated node to eventually find a better match
					for( std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* >::iterator nit = neighbors.begin(); nit != neighbors.end(); ++nit ) {

						spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_src = *nit;

						if( !n_src )
							continue;

						if( n->value_.border_ != n_src->value_.border_ )
							continue;

						// find matching surfel for the view direction, but allow to use a slightly worse fit,
						// when it is the only one with sufficient points for matching
						MultiResolutionColorSurfelMap::Surfel* bestMatchSurfel = NULL;
						int bestMatchSurfelIdx = -1;
						double bestMatchDist = -1.f;
						for( unsigned int k = 0; k < MAX_NUM_SURFELS; k++ ) {

							const MultiResolutionColorSurfelMap::Surfel& srcSurfel = n_src->value_.surfels_[k];

							if( srcSurfel.num_points_ < MIN_SURFEL_POINTS )
								continue;

							const double dist = dir_match_src.dot( srcSurfel.initial_view_dir_ );
							if( dist >= params_.surfel_match_angle_threshold_ && dist >= bestMatchDist ) {
								bestMatchSurfel = &n_src->value_.surfels_[k];
								bestMatchDist = dist;
								bestMatchSurfelIdx = k;
							}
						}

						if( !bestMatchSurfel ) {
							continue;
						}

						// calculate error metric for matching surfels
						double dist_pos2 = (bestMatchSurfel->mean_.block<3,1>(0,0) - pos_match_src.block<3,1>(0,0)).squaredNorm();

						if( dist_pos2 > search_dist2_ )
							continue;

						float featureDist = 0.f;
						if( use_features_) {
							featureDist = surfel.agglomerated_shape_texture_features_.distance( bestMatchSurfel->agglomerated_shape_texture_features_ );
							if( featureDist > params_.max_feature_dist2_ )
								continue;
						}

						if( params_.registration_use_color_ ) {
							Eigen::Matrix< double, 3, 1 > diff;
							diff = bestMatchSurfel->mean_.block<3,1>(3,0) - surfel.mean_.block<3,1>(3,0);
							if( fabs(diff(0)) > params_.luminance_reg_threshold_ )
								continue;
							if( fabs(diff(1)) > params_.color_reg_threshold_ )
								continue;
							if( fabs(diff(2)) > params_.color_reg_threshold_ )
								continue;
						}

						float assocDist = sqrtf(dist_pos2);

						if( use_features_ )
							assocDist *= featureDist;

						if( assocDist < bestAssocDist ) {
							bestAssocDist = assocDist;
							bestAssocFeatureDist = featureDist;
							n->value_.surfels_[i].assocDist_ = assocDist;

							bestAssoc.n_src_ = n;
							bestAssoc.src_ = &n->value_.surfels_[i];
							bestAssoc.src_idx_ = i;
							bestAssoc.n_dst_ = n_src;
							bestAssoc.dst_ = bestMatchSurfel;
							bestAssoc.dst_idx_ = bestMatchSurfelIdx;
							bestAssoc.match = 1;

							if( use_features_ )
								bestAssoc.weight = params_.max_feature_dist2_ - featureDist;
							else
								bestAssoc.weight = (1+numChildren) * 1.f;
						}

					}

				}

			}

		}

		if( bestAssocDist != std::numeric_limits<float>::max() ) {


//			bestAssoc.weight *= n->value_.assocWeight_;
//			bestAssoc.weight = 1.f;

			associations_->push_back( bestAssoc );
			n->value_.association_ = bestAssoc.n_dst_;
			n->value_.associated_ = 1;
			n->value_.assocSurfelIdx_ = bestAssoc.src_idx_;
			n->value_.assocSurfelDstIdx_ = bestAssoc.dst_idx_;
		}
		else {
			n->value_.association_ = NULL;
			n->value_.associated_ = 0;
		}


	}


	tbb::concurrent_vector< MultiResolutionColorSurfelRegistration::SurfelAssociation >* associations_;
	MultiResolutionColorSurfelRegistration::Params params_;
	MultiResolutionColorSurfelMap* source_;
	MultiResolutionColorSurfelMap* target_;
	std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* >* nodes_;
	Eigen::Matrix4d transform_;
	Eigen::Matrix4f transformf_;
	Eigen::Matrix3d rotation_;
	int process_depth_;
	float process_resolution_, search_dist_, search_dist2_;
	Eigen::Vector4f search_dist_vec_;
	bool use_features_;
	int num_vol_queries_, num_finds_, num_neighbors_;

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};


void MultiResolutionColorSurfelRegistration::associateNodeListParallel( MultiResolutionColorSurfelRegistration::SurfelAssociationList& surfelAssociations, MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* >& nodes, int processDepth, Eigen::Matrix4d& transform, double searchDistFactor, double maxSearchDist, bool useFeatures ) {

	tbb::concurrent_vector< MultiResolutionColorSurfelRegistration::SurfelAssociation > depthAssociations;
	depthAssociations.reserve( nodes.size() );

	// only process nodes that are active (should improve parallel processing)
	std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* > activeNodes;
	activeNodes.reserve( nodes.size() );

	for( unsigned int i = 0; i < nodes.size(); i++ ) {

		spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n = nodes[i];

		if( n->value_.associated_ == -1 )
			continue;

		activeNodes.push_back( n );

	}



	AssociateFunctor af( &depthAssociations, params_, &source, &target, &activeNodes, transform, processDepth, searchDistFactor, maxSearchDist, useFeatures );

	if( params_.parallel_ )
		tbb::parallel_for_each( activeNodes.begin(), activeNodes.end(), af );
	else
		std::for_each( activeNodes.begin(), activeNodes.end(), af );


	surfelAssociations.insert( surfelAssociations.end(), depthAssociations.begin(), depthAssociations.end() );

}


void MultiResolutionColorSurfelRegistration::associatePointFeatures() {

	pcl::StopWatch sw;
	sw.reset();

	const int numNeighbors = params_.pointFeatureMatchingNumNeighbors_;
	const int distanceThreshold = params_.pointFeatureMatchingThreshold_;

	featureAssociations_.clear();

	if (!target_->lsh_index_ || !source_->lsh_index_)
		return;

	featureAssociations_.reserve(std::min(source_->features_.size(), target_->features_.size()));

	// find associations from source to target
	// build up query matrix
	flann::Matrix<unsigned char> sourceQuery(source_->descriptors_.data, source_->descriptors_.rows, source_->descriptors_.cols);
	flann::Matrix<unsigned char> targetQuery(target_->descriptors_.data, target_->descriptors_.rows, target_->descriptors_.cols);

	// indices in source features for target features
	flann::Matrix<int> sourceIndices(new int[targetQuery.rows * numNeighbors], targetQuery.rows, numNeighbors);
	flann::Matrix<int> sourceDists(new int[targetQuery.rows * numNeighbors], targetQuery.rows, numNeighbors);

	// indices in target features for source features
	flann::Matrix<int> targetIndices(new int[sourceQuery.rows * numNeighbors], sourceQuery.rows, numNeighbors);
	flann::Matrix<int> targetDists(new int[sourceQuery.rows * numNeighbors], sourceQuery.rows, numNeighbors);

	target_->lsh_index_->knnSearch(sourceQuery, targetIndices, targetDists, numNeighbors, flann::SearchParams());
	source_->lsh_index_->knnSearch(targetQuery, sourceIndices, sourceDists, numNeighbors, flann::SearchParams());

	if( params_.debugFeatures_ )
		std::cout << "flann query took: " << sw.getTime() << "\n";
	sw.reset();

	// find mutually consistent matches within distance threshold
	for (unsigned int i = 0; i < sourceQuery.rows; i++) {

		// check if source feature is among nearest neighbors of matched target feature
		for (unsigned int n = 0; n < numNeighbors; n++) {

			if (targetDists.ptr()[i * numNeighbors + n] > distanceThreshold)
				continue;

			int targetIdx = targetIndices.ptr()[i * numNeighbors + n];

			if (targetIdx < 0 || targetIdx >= sourceIndices.rows)
				continue;

			for (unsigned int n2 = 0; n2 < numNeighbors; n2++) {

				if (sourceDists.ptr()[targetIdx * numNeighbors + n2] > distanceThreshold)
					continue;

				int sourceIdx = sourceIndices.ptr()[targetIdx * numNeighbors + n2];

				if (sourceIdx < 0 || sourceIdx >= targetIndices.rows)
					continue;

				if (sourceIdx == i) {
					MultiResolutionColorSurfelRegistration::FeatureAssociation assoc( i, targetIdx);
					featureAssociations_.push_back(assoc);
//		    			consistentMatches.push_back( std::pair< int, int >( i, targetIdx ) );
					break;
				}

			}

		}

	}

	if( params_.debugFeatures_ )
		std::cout << "consistent match search took: " << sw.getTime() << "\n";


	delete[] sourceIndices.ptr();
	delete[] targetIndices.ptr();
	delete[] sourceDists.ptr();
	delete[] targetDists.ptr();

}


class GradientFunctor {
public:
	GradientFunctor( MultiResolutionColorSurfelRegistration::SurfelAssociationList* assocList, const MultiResolutionColorSurfelRegistration::Params& params, double tx, double ty, double tz, double qx, double qy, double qz, double qw, bool relativeDerivatives, bool deriv2 = false, bool interpolate_neighbors = true, bool derivZ = false ) {

		assocList_ = assocList;

		params_ = params;

		const double inv_qw = 1.0 / qw;

		relativeDerivatives_ = relativeDerivatives;
		deriv2_ = deriv2;
		derivZ_ = derivZ;
		interpolate_neighbors_ = interpolate_neighbors;

		currentTransform.setIdentity();
		currentTransform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
		currentTransform(0,3) = tx;
		currentTransform(1,3) = ty;
		currentTransform(2,3) = tz;


//		cov_cc_add.setIdentity();
//		cov_cc_add *= SMOOTH_COLOR_COVARIANCE;

		currentRotation = Eigen::Matrix3d( currentTransform.block<3,3>(0,0) );
		currentRotationT = currentRotation.transpose();
		currentTranslation = Eigen::Vector3d( currentTransform.block<3,1>(0,3) );


		// build up derivatives of rotation and translation for the transformation variables
		dt_tx(0) = 1.f; dt_tx(1) = 0.f; dt_tx(2) = 0.f;
		dt_ty(0) = 0.f; dt_ty(1) = 1.f; dt_ty(2) = 0.f;
		dt_tz(0) = 0.f; dt_tz(1) = 0.f; dt_tz(2) = 1.f;


		if( relativeDerivatives_ ) {

			dR_qx.setZero();
			dR_qx(1,2) = -2;
			dR_qx(2,1) = 2;

//			dR_qx = (dR_qx * currentRotation).eval();


			dR_qy.setZero();
			dR_qy(0,2) = 2;
			dR_qy(2,0) = -2;

//			dR_qy = (dR_qy * currentRotation).eval();


			dR_qz.setZero();
			dR_qz(0,1) = -2;
			dR_qz(1,0) = 2;

//			dR_qz = (dR_qz * currentRotation).eval();

		}
		else {

			// matrix(
			//  [ 0,
			//    2*((qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qy),
			//    2*(qz-(qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)) ],
			//  [ 2*(qy-(qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)),
			//    -4*qx,
			//    2*(qx^2/sqrt(-qz^2-qy^2-qx^2+1)-sqrt(-qz^2-qy^2-qx^2+1)) ],
			//  [ 2*((qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)+qz),
			//    2*(sqrt(-qz^2-qy^2-qx^2+1)-qx^2/sqrt(-qz^2-qy^2-qx^2+1)),
			//    -4*qx ]
			// )
			dR_qx(0,0) = 0.0;
			dR_qx(0,1) = 2.0*((qx*qz)*inv_qw+qy);
			dR_qx(0,2) = 2.0*(qz-(qx*qy)*inv_qw);
			dR_qx(1,0) = 2.0*(qy-(qx*qz)*inv_qw);
			dR_qx(1,1) = -4.0*qx;
			dR_qx(1,2) = 2.0*(qx*qx*inv_qw-qw);
			dR_qx(2,0) = 2.0*((qx*qy)*inv_qw+qz);
			dR_qx(2,1) = 2.0*(qw-qx*qx*inv_qw);
			dR_qx(2,2) = -4.0*qx;

			// matrix(
			//  [ -4*qy,
			//    2*((qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qx),
			//    2*(sqrt(-qz^2-qy^2-qx^2+1)-qy^2/sqrt(-qz^2-qy^2-qx^2+1)) ],
			//  [ 2*(qx-(qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)),
			//    0,
			//    2*((qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)+qz) ],
			//  [ 2*(qy^2/sqrt(-qz^2-qy^2-qx^2+1)-sqrt(-qz^2-qy^2-qx^2+1)),
			//    2*(qz-(qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)),
			//    -4*qy ]
			// )

			dR_qy(0,0) = -4.0*qy;
			dR_qy(0,1) = 2.0*((qy*qz)*inv_qw+qx);
			dR_qy(0,2) = 2.0*(qw-qy*qy*inv_qw);
			dR_qy(1,0) = 2.0*(qx-(qy*qz)*inv_qw);
			dR_qy(1,1) = 0.0;
			dR_qy(1,2) = 2.0*((qx*qy)*inv_qw+qz);
			dR_qy(2,0) = 2.0*(qy*qy*inv_qw-qw);
			dR_qy(2,1) = 2.0*(qz-(qx*qy)*inv_qw);
			dR_qy(2,2) = -4.0*qy;

			// matrix(
			//  [ -4*qz,
			//    2*(qz^2/sqrt(-qz^2-qy^2-qx^2+1)-sqrt(-qz^2-qy^2-qx^2+1)),
			//    2*(qx-(qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)) ],
			//  [ 2*(sqrt(-qz^2-qy^2-qx^2+1)-qz^2/sqrt(-qz^2-qy^2-qx^2+1)),
			//    -4*qz,
			//    2*((qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qy) ],
			//  [ 2*((qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qx),
			//    2*(qy-(qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)),
			//    0 ]
			// )
			dR_qz(0,0) = -4.0*qz;
			dR_qz(0,1) = 2.0*(qz*qz*inv_qw-qw);
			dR_qz(0,2) = 2.0*(qx-(qy*qz)*inv_qw);
			dR_qz(1,0) = 2.0*(qw-qz*qz*inv_qw);
			dR_qz(1,1) = -4.0*qz;
			dR_qz(1,2) = 2.0*((qx*qz)*inv_qw+qy);
			dR_qz(2,0) = 2.0*((qy*qz)*inv_qw+qx);
			dR_qz(2,1) = 2.0*(qy-(qx*qz)*inv_qw);
			dR_qz(2,2) = 0.0;

		}


		dR_qxT = dR_qx.transpose();
		dR_qyT = dR_qy.transpose();
		dR_qzT = dR_qz.transpose();


		ddiff_s_tx.block<3,1>(0,0) = -dt_tx;
		ddiff_s_ty.block<3,1>(0,0) = -dt_ty;
		ddiff_s_tz.block<3,1>(0,0) = -dt_tz;

		if( deriv2_ ) {

			if( relativeDerivatives_ ) {

				d2R_qxx( 0, 0 ) = 0;
				d2R_qxx( 0, 1 ) = 0;
				d2R_qxx( 0, 2 ) = 0;
				d2R_qxx( 1, 0 ) = 0;
				d2R_qxx( 1, 1 ) = -4.0;
				d2R_qxx( 1, 2 ) = 0;
				d2R_qxx( 2, 0 ) = 0;
				d2R_qxx( 2, 1 ) = 0;
				d2R_qxx( 2, 2 ) = -4.0;

//				d2R_qxx = (d2R_qxx * currentRotation).eval();


				d2R_qxy( 0, 0 ) = 0.0;
				d2R_qxy( 0, 1 ) = 2;
				d2R_qxy( 0, 2 ) = 0;
				d2R_qxy( 1, 0 ) = 2;
				d2R_qxy( 1, 1 ) = 0.0;
				d2R_qxy( 1, 2 ) = 0;
				d2R_qxy( 2, 0 ) = 0;
				d2R_qxy( 2, 1 ) = 0;
				d2R_qxy( 2, 2 ) = 0.0;

//				d2R_qxy = (d2R_qxy * currentRotation).eval();


				d2R_qxz( 0, 0 ) = 0.0;
				d2R_qxz( 0, 1 ) = 0;
				d2R_qxz( 0, 2 ) = 2;
				d2R_qxz( 1, 0 ) = 0;
				d2R_qxz( 1, 1 ) = 0.0;
				d2R_qxz( 1, 2 ) = 0;
				d2R_qxz( 2, 0 ) = 2;
				d2R_qxz( 2, 1 ) = 0;
				d2R_qxz( 2, 2 ) = 0.0;

//				d2R_qxz = (d2R_qxz * currentRotation).eval();


				d2R_qyy( 0, 0 ) = -4.0;
				d2R_qyy( 0, 1 ) = 0;
				d2R_qyy( 0, 2 ) = 0;
				d2R_qyy( 1, 0 ) = 0;
				d2R_qyy( 1, 1 ) = 0.0;
				d2R_qyy( 1, 2 ) = 0;
				d2R_qyy( 2, 0 ) = 0;
				d2R_qyy( 2, 1 ) = 0;
				d2R_qyy( 2, 2 ) = -4.0;

//				d2R_qyy = (d2R_qyy * currentRotation).eval();


				d2R_qyz( 0, 0 ) = 0.0;
				d2R_qyz( 0, 1 ) = 0;
				d2R_qyz( 0, 2 ) = 0;
				d2R_qyz( 1, 0 ) = 0;
				d2R_qyz( 1, 1 ) = 0.0;
				d2R_qyz( 1, 2 ) = 2;
				d2R_qyz( 2, 0 ) = 0;
				d2R_qyz( 2, 1 ) = 2;
				d2R_qyz( 2, 2 ) = 0.0;

//				d2R_qyz = (d2R_qyz * currentRotation).eval();


				d2R_qzz( 0, 0 ) = -4.0;
				d2R_qzz( 0, 1 ) = 0;
				d2R_qzz( 0, 2 ) = 0;
				d2R_qzz( 1, 0 ) = 0;
				d2R_qzz( 1, 1 ) = -4.0;
				d2R_qzz( 1, 2 ) = 0;
				d2R_qzz( 2, 0 ) = 0;
				d2R_qzz( 2, 1 ) = 0;
				d2R_qzz( 2, 2 ) = 0.0;

//				d2R_qzz = (d2R_qzz * currentRotation).eval();


			}
			else {

				const double inv_qw3 = inv_qw*inv_qw*inv_qw;

				// matrix(
				// [ 0,
				//   2*(qz/sqrt(-qz^2-qy^2-qx^2+1)+(qx^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-qy/sqrt(-qz^2-qy^2-qx^2+1)-(qx^2*qy)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(-qz/sqrt(-qz^2-qy^2-qx^2+1)-(qx^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   -4,
				//   2*((3*qx)/sqrt(-qz^2-qy^2-qx^2+1)+qx^3/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(qy/sqrt(-qz^2-qy^2-qx^2+1)+(qx^2*qy)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-(3*qx)/sqrt(-qz^2-qy^2-qx^2+1)-qx^3/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   -4 ] )
				d2R_qxx( 0, 0 ) = 0;
				d2R_qxx( 0, 1 ) = 2.0*(qz*inv_qw+qx*qx*qz*inv_qw3);
				d2R_qxx( 0, 2 ) = 2.0*(-qy*inv_qw-qx*qx*qy*inv_qw3);
				d2R_qxx( 1, 0 ) = 2.0*(-qz*inv_qw-qx*qx*qz*inv_qw3);
				d2R_qxx( 1, 1 ) = -4.0;
				d2R_qxx( 1, 2 ) = 2.0*(3.0*qx*inv_qw+qx*qx*qx*inv_qw3);
				d2R_qxx( 2, 0 ) = 2.0*(qy*inv_qw+qx*qx*qy*inv_qw3);
				d2R_qxx( 2, 1 ) = 2.0*(-3.0*qx*inv_qw-qx*qx*qx*inv_qw3);
				d2R_qxx( 2, 2 ) = -4.0;


				// matrix(
				// [ 0,
				//   2*((qx*qy*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)+1),
				//   2*(-qx/sqrt(-qz^2-qy^2-qx^2+1)-(qx*qy^2)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(1-(qx*qy*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   0,
				//   2*(qy/sqrt(-qz^2-qy^2-qx^2+1)+(qx^2*qy)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(qx/sqrt(-qz^2-qy^2-qx^2+1)+(qx*qy^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-qy/sqrt(-qz^2-qy^2-qx^2+1)-(qx^2*qy)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   0 ] )
				d2R_qxy( 0, 0 ) = 0.0;
				d2R_qxy( 0, 1 ) = 2.0*(qx*qy*qz*inv_qw3+1.0);
				d2R_qxy( 0, 2 ) = 2.0*(-qx*inv_qw-qx*qy*qy*inv_qw3);
				d2R_qxy( 1, 0 ) = 2.0*(1.0-qx*qy*qz*inv_qw3);
				d2R_qxy( 1, 1 ) = 0.0;
				d2R_qxy( 1, 2 ) = 2.0*(qy*inv_qw+qx*qx*qy*inv_qw3);
				d2R_qxy( 2, 0 ) = 2.0*(qx*inv_qw+qx*qy*qy*inv_qw3);
				d2R_qxy( 2, 1 ) = 2.0*(-qy*inv_qw-qx*qx*qy*inv_qw3);
				d2R_qxy( 2, 2 ) = 0.0;


				// matrix(
				// [ 0,
				//   2*(qx/sqrt(-qz^2-qy^2-qx^2+1)+(qx*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(1-(qx*qy*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(-qx/sqrt(-qz^2-qy^2-qx^2+1)-(qx*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   0,
				//   2*(qz/sqrt(-qz^2-qy^2-qx^2+1)+(qx^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*((qx*qy*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)+1),
				//   2*(-qz/sqrt(-qz^2-qy^2-qx^2+1)-(qx^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				// 0 ])
				d2R_qxz( 0, 0 ) = 0.0;
				d2R_qxz( 0, 1 ) = 2.0*(qx*inv_qw+qx*qz*qz*inv_qw3);
				d2R_qxz( 0, 2 ) = 2.0*(1.0-qx*qy*qz*inv_qw3);
				d2R_qxz( 1, 0 ) = 2.0*(-qx*inv_qw-qx*qz*qz*inv_qw3);
				d2R_qxz( 1, 1 ) = 0.0;
				d2R_qxz( 1, 2 ) = 2.0*(qz*inv_qw+qx*qx*qz*inv_qw3);
				d2R_qxz( 2, 0 ) = 2.0*(qx*qy*qz*inv_qw3+1.0);
				d2R_qxz( 2, 1 ) = 2.0*(-qz*inv_qw-qx*qx*qz*inv_qw3);
				d2R_qxz( 2, 2 ) = 0.0;

				// matrix(
				// [ -4,
				//   2*(qz/sqrt(-qz^2-qy^2-qx^2+1)+(qy^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-(3*qy)/sqrt(-qz^2-qy^2-qx^2+1)-qy^3/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(-qz/sqrt(-qz^2-qy^2-qx^2+1)-(qy^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   0,
				//   2*(qx/sqrt(-qz^2-qy^2-qx^2+1)+(qx*qy^2)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*((3*qy)/sqrt(-qz^2-qy^2-qx^2+1)+qy^3/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-qx/sqrt(-qz^2-qy^2-qx^2+1)-(qx*qy^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   -4 ])
				d2R_qyy( 0, 0 ) = -4.0;
				d2R_qyy( 0, 1 ) = 2.0*(qz*inv_qw+qy*qy*qz*inv_qw3);
				d2R_qyy( 0, 2 ) = 2.0*(-3.0*qy*inv_qw-qy*qy*qy*inv_qw3);
				d2R_qyy( 1, 0 ) = 2.0*(-qz*inv_qw-qy*qy*qz*inv_qw3);
				d2R_qyy( 1, 1 ) = 0.0;
				d2R_qyy( 1, 2 ) = 2.0*(qx*inv_qw+qx*qy*qy*inv_qw3);
				d2R_qyy( 2, 0 ) = 2.0*(3.0*qy*inv_qw+qy*qy*qy*inv_qw3);
				d2R_qyy( 2, 1 ) = 2.0*(-qx*inv_qw-qx*qy*qy*inv_qw3);
				d2R_qyy( 2, 2 ) = -4.0;

				// matrix(
				// [ 0,
				//   2*(qy/sqrt(-qz^2-qy^2-qx^2+1)+(qy*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-qz/sqrt(-qz^2-qy^2-qx^2+1)-(qy^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(-qy/sqrt(-qz^2-qy^2-qx^2+1)-(qy*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   0,
				//   2*((qx*qy*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)+1) ],
				// [ 2*(qz/sqrt(-qz^2-qy^2-qx^2+1)+(qy^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(1-(qx*qy*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   0 ])
				d2R_qyz( 0, 0 ) = 0.0;
				d2R_qyz( 0, 1 ) = 2.0*(qy*inv_qw+qy*qz*qz*inv_qw3);
				d2R_qyz( 0, 2 ) = 2.0*(-qz*inv_qw-qy*qy*qz*inv_qw3);
				d2R_qyz( 1, 0 ) = 2.0*(-qy*inv_qw-qy*qz*qz*inv_qw3);
				d2R_qyz( 1, 1 ) = 0.0;
				d2R_qyz( 1, 2 ) = 2.0*(qx*qy*qz*inv_qw3+1.0);
				d2R_qyz( 2, 0 ) = 2.0*(qz*inv_qw+qy*qy*qz*inv_qw3);
				d2R_qyz( 2, 1 ) = 2.0*(1.0-qx*qy*qz*inv_qw3);
				d2R_qyz( 2, 2 ) = 0.0;

				// matrix(
				// [ -4,
				//   2*((3*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qz^3/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-qy/sqrt(-qz^2-qy^2-qx^2+1)-(qy*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(-(3*qz)/sqrt(-qz^2-qy^2-qx^2+1)-qz^3/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   -4,
				//   2*(qx/sqrt(-qz^2-qy^2-qx^2+1)+(qx*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(qy/sqrt(-qz^2-qy^2-qx^2+1)+(qy*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-qx/sqrt(-qz^2-qy^2-qx^2+1)-(qx*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   0 ])
				d2R_qzz( 0, 0 ) = -4.0;
				d2R_qzz( 0, 1 ) = 2.0*(3.0*qz*inv_qw+qz*qz*qz*inv_qw3);
				d2R_qzz( 0, 2 ) = 2.0*(-qy*inv_qw-qy*qz*qz*inv_qw3);
				d2R_qzz( 1, 0 ) = 2.0*(-3.0*qz*inv_qw-qz*qz*qz*inv_qw3);
				d2R_qzz( 1, 1 ) = -4.0;
				d2R_qzz( 1, 2 ) = 2.0*(qx*inv_qw+qx*qz*qz*inv_qw3);
				d2R_qzz( 2, 0 ) = 2.0*(qy*inv_qw+qy*qz*qz*inv_qw3);
				d2R_qzz( 2, 1 ) = 2.0*(-qx*inv_qw-qx*qz*qz*inv_qw3);
				d2R_qzz( 2, 2 ) = 0.0;

			}

			d2R_qxxT = d2R_qxx.transpose();
			d2R_qxyT = d2R_qxy.transpose();
			d2R_qxzT = d2R_qxz.transpose();
			d2R_qyyT = d2R_qyy.transpose();
			d2R_qyzT = d2R_qyz.transpose();
			d2R_qzzT = d2R_qzz.transpose();


			if( derivZ_ ) {

				// needed for the derivatives for the measurements

				ddiff_dzmx = Eigen::Vector3d( 1.0, 0.0, 0.0 );
				ddiff_dzmy = Eigen::Vector3d( 0.0, 1.0, 0.0 );
				ddiff_dzmz = Eigen::Vector3d( 0.0, 0.0, 1.0 );

				ddiff_dzsx = -currentRotation * Eigen::Vector3d( 1.0, 0.0, 0.0 );
				ddiff_dzsy = -currentRotation * Eigen::Vector3d( 0.0, 1.0, 0.0 );
				ddiff_dzsz = -currentRotation * Eigen::Vector3d( 0.0, 0.0, 1.0 );

				d2diff_qx_zsx = -dR_qx * Eigen::Vector3d( 1.0, 0.0, 0.0 );
				d2diff_qx_zsy = -dR_qx * Eigen::Vector3d( 0.0, 1.0, 0.0 );
				d2diff_qx_zsz = -dR_qx * Eigen::Vector3d( 0.0, 0.0, 1.0 );
				d2diff_qy_zsx = -dR_qy * Eigen::Vector3d( 1.0, 0.0, 0.0 );
				d2diff_qy_zsy = -dR_qy * Eigen::Vector3d( 0.0, 1.0, 0.0 );
				d2diff_qy_zsz = -dR_qy * Eigen::Vector3d( 0.0, 0.0, 1.0 );
				d2diff_qz_zsx = -dR_qz * Eigen::Vector3d( 1.0, 0.0, 0.0 );
				d2diff_qz_zsy = -dR_qz * Eigen::Vector3d( 0.0, 1.0, 0.0 );
				d2diff_qz_zsz = -dR_qz * Eigen::Vector3d( 0.0, 0.0, 1.0 );

			}

		}

	}

	~GradientFunctor() {}


	void operator()( const tbb::blocked_range<size_t>& r ) const {
		for( size_t i=r.begin(); i!=r.end(); ++i )
			(*this)((*assocList_)[i]);
	}



	void operator()( MultiResolutionColorSurfelRegistration::SurfelAssociation& assoc ) const {


		if( assoc.match == 0 || !assoc.src_->applyUpdate_ || !assoc.dst_->applyUpdate_ ) {
			assoc.match = 0;
			return;
		}

		const float processResolution = assoc.n_src_->resolution();
		double weight = assoc.weight;

		Eigen::Vector4d pos;
		pos.block<3,1>(0,0) = assoc.src_->mean_.block<3,1>(0,0);
		pos(3,0) = 1.f;

		const Eigen::Vector4d pos_src = currentTransform * pos;

		double error = 0;

		double de_tx = 0;
		double de_ty = 0;
		double de_tz = 0;
		double de_qx = 0;
		double de_qy = 0;
		double de_qz = 0;

		Eigen::Matrix< double, 6, 6 > d2J_pp;
		Eigen::Matrix< double, 6, 6 > JSzJ;
		if( deriv2_ ) {
			d2J_pp.setZero();
			JSzJ.setZero();
		}



		// spatial component, marginalized

		Eigen::Matrix3d cov_ss_add;
		cov_ss_add.setZero();
		if( params_.add_smooth_pos_covariance_ ) {
			cov_ss_add.setIdentity();
			cov_ss_add *= params_.smooth_surface_cov_factor_ * processResolution*processResolution;
		}

		Eigen::Matrix3d cov1_ss;
		Eigen::Matrix3d cov2_ss = assoc.src_->cov_.block<3,3>(0,0) + cov_ss_add;

		Eigen::Vector3d dstMean;
		Eigen::Vector3d srcMean = assoc.src_->mean_.block<3,1>(0,0);

		bool in_interpolation_range = false;

		if( interpolate_neighbors_ ) {

			// use trilinear interpolation to handle discretization effects
			// => associate with neighbors and weight correspondences
			// only makes sense when match is within resolution distance to the node center
			const float resolution = processResolution;
			Eigen::Vector3d centerDiff = assoc.n_dst_->getCenterPosition().block<3,1>(0,0).cast<double>() - pos_src.block<3,1>(0,0);
			if( resolution - fabsf(centerDiff(0)) > 0  && resolution - fabsf(centerDiff(1)) > 0  && resolution - fabsf(centerDiff(2)) > 0 ) {

				in_interpolation_range = true;

				// associate with neighbors for which distance to the node center is smaller than resolution

				dstMean.setZero();
				cov1_ss.setZero();

				double sumWeight = 0.f;
				double sumWeight2 = 0.f;

				for( int s = 0; s < 27; s++ ) {

					spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_dst_n = assoc.n_dst_->neighbors_[s];

					if(!n_dst_n)
						continue;

					MultiResolutionColorSurfelMap::Surfel* dst_n = &n_dst_n->value_.surfels_[assoc.dst_idx_];
					if( dst_n->num_points_ < MIN_SURFEL_POINTS )
						continue;

					Eigen::Vector3d centerDiff_n = n_dst_n->getCenterPosition().block<3,1>(0,0).cast<double>() - pos_src.block<3,1>(0,0);
					const double dx = resolution - fabsf(centerDiff_n(0));
					const double dy = resolution - fabsf(centerDiff_n(1));
					const double dz = resolution - fabsf(centerDiff_n(2));

					if( dx > 0 && dy > 0 && dz > 0 ) {

						const double weight = dx*dy*dz;

						dstMean += weight * dst_n->mean_.block<3,1>(0,0);
						cov1_ss += weight*weight * (dst_n->cov_.block<3,3>(0,0));

						sumWeight += weight;
						sumWeight2 += weight*weight;

					}


				}

				// numerically stable?
				if( sumWeight > resolution* 1e-6 ) {
					dstMean /= sumWeight;
					cov1_ss /= sumWeight2;

				}
				else
					in_interpolation_range = false;

				cov1_ss += cov_ss_add;


			}

		}

		if( !interpolate_neighbors_ || !in_interpolation_range ) {

			dstMean = assoc.dst_->mean_.block<3,1>(0,0);
			cov1_ss = assoc.dst_->cov_.block<3,3>(0,0) + cov_ss_add;

		}


		// has only marginal (positive!) effect on visual odometry result
		// makes tracking more robust (when only few surfels available)
		cov1_ss *= INTERPOLATION_COV_FACTOR;
		cov2_ss *= INTERPOLATION_COV_FACTOR;

		const Eigen::Vector3d TsrcMean = pos_src.block<3,1>(0,0);
		const Eigen::Vector3d diff_s = dstMean - TsrcMean;

		const Eigen::Matrix3d Rcov2_ss = currentRotation * cov2_ss;
		const Eigen::Matrix3d Rcov2_ssT = Rcov2_ss.transpose();

		const Eigen::Matrix3d cov_ss = cov1_ss + Rcov2_ss * currentRotationT;
		const Eigen::Matrix3d invcov_ss = cov_ss.inverse();
		const Eigen::Vector3d invcov_ss_diff_s = invcov_ss * diff_s;

		error = log( cov_ss.determinant() ) + diff_s.dot(invcov_ss_diff_s);


		if( relativeDerivatives_ ) {

//			const Eigen::Matrix3d dRR_qx = dR_qx * currentRotation;
//			const Eigen::Matrix3d dRR_qy = dR_qy * currentRotation;
//			const Eigen::Matrix3d dRR_qz = dR_qz * currentRotation;
//
//			const Eigen::Matrix3d d2RR_qxx = d2R_qxx * currentRotation;
//			const Eigen::Matrix3d d2RR_qxy = d2R_qxy * currentRotation;
//			const Eigen::Matrix3d d2RR_qxz = d2R_qxz * currentRotation;
//			const Eigen::Matrix3d d2RR_qyy = d2R_qyy * currentRotation;
//			const Eigen::Matrix3d d2RR_qyz = d2R_qyz * currentRotation;
//			const Eigen::Matrix3d d2RR_qzz = d2R_qzz * currentRotation;

			const Eigen::Matrix3d Rcov2R_ss = Rcov2_ss * currentRotationT;

			const Eigen::Vector3d ddiff_s_qx = -dR_qx * TsrcMean;
			const Eigen::Vector3d ddiff_s_qy = -dR_qy * TsrcMean;
			const Eigen::Vector3d ddiff_s_qz = -dR_qz * TsrcMean;



			const Eigen::Matrix3d dcov_ss_qx = dR_qx * Rcov2R_ss + Rcov2R_ss * dR_qx.transpose();
			const Eigen::Matrix3d dcov_ss_qy = dR_qy * Rcov2R_ss + Rcov2R_ss * dR_qy.transpose();
			const Eigen::Matrix3d dcov_ss_qz = dR_qz * Rcov2R_ss + Rcov2R_ss * dR_qz.transpose();

			const Eigen::Matrix3d dinvcov_ss_qx = -invcov_ss * dcov_ss_qx * invcov_ss;
			const Eigen::Matrix3d dinvcov_ss_qy = -invcov_ss * dcov_ss_qy * invcov_ss;
			const Eigen::Matrix3d dinvcov_ss_qz = -invcov_ss * dcov_ss_qz * invcov_ss;

			const Eigen::Vector3d dinvcov_ss_qx_diff_s = dinvcov_ss_qx * diff_s;
			const Eigen::Vector3d dinvcov_ss_qy_diff_s = dinvcov_ss_qy * diff_s;
			const Eigen::Vector3d dinvcov_ss_qz_diff_s = dinvcov_ss_qz * diff_s;


			de_tx = 2.0 * ddiff_s_tx.dot(invcov_ss_diff_s);
			de_ty = 2.0 * ddiff_s_ty.dot(invcov_ss_diff_s);
			de_tz = 2.0 * ddiff_s_tz.dot(invcov_ss_diff_s);
			de_qx = 2.0 * ddiff_s_qx.dot(invcov_ss_diff_s) + diff_s.dot(dinvcov_ss_qx_diff_s);
			de_qy = 2.0 * ddiff_s_qy.dot(invcov_ss_diff_s) + diff_s.dot(dinvcov_ss_qy_diff_s);
			de_qz = 2.0 * ddiff_s_qz.dot(invcov_ss_diff_s) + diff_s.dot(dinvcov_ss_qz_diff_s);

			// second term: derivative for normalizer of the normal distribution! det(cov) is not independent of q!
			// -log( (2pi)^-(3/2) (det(cov))^(-1/2) )
			// = - log( (2pi)^-(3/2) ) - log( (det(cov))^(-1/2) )
			// = const. - (-0.5) * log( det(cov) )
			// = 0.5 * log( det(cov) ) => 0.5 factor can be left out also for the exp part...
			// d(log(det(cov)))/dq = 1/det(cov) * det(cov) * tr( cov^-1 * dcov/dq )
			// = tr( cov^-1 * dcov/dq )
			de_qx += (invcov_ss * dcov_ss_qx).trace();
			de_qy += (invcov_ss * dcov_ss_qy).trace();
			de_qz += (invcov_ss * dcov_ss_qz).trace();


			if( deriv2_ ) {

				const Eigen::Vector3d d2diff_s_qxx = -d2R_qxx * TsrcMean;
				const Eigen::Vector3d d2diff_s_qxy = -d2R_qxy * TsrcMean;
				const Eigen::Vector3d d2diff_s_qxz = -d2R_qxz * TsrcMean;
				const Eigen::Vector3d d2diff_s_qyy = -d2R_qyy * TsrcMean;
				const Eigen::Vector3d d2diff_s_qyz = -d2R_qyz * TsrcMean;
				const Eigen::Vector3d d2diff_s_qzz = -d2R_qzz * TsrcMean;

				const Eigen::Matrix3d d2cov_ss_qxx = d2R_qxx * Rcov2R_ss + 2.0 * dR_qx * Rcov2R_ss * dR_qxT + Rcov2R_ss * d2R_qxxT;
				const Eigen::Matrix3d d2cov_ss_qxy = d2R_qxy * Rcov2R_ss + dR_qx * Rcov2R_ss * dR_qyT + dR_qy * Rcov2R_ss * dR_qxT + Rcov2R_ss * d2R_qxyT;
				const Eigen::Matrix3d d2cov_ss_qxz = d2R_qxz * Rcov2R_ss + dR_qx * Rcov2R_ss * dR_qzT + dR_qz * Rcov2R_ss * dR_qxT + Rcov2R_ss * d2R_qxzT;
				const Eigen::Matrix3d d2cov_ss_qyy = d2R_qyy * Rcov2R_ss + 2.0 * dR_qy * Rcov2R_ss * dR_qyT + Rcov2R_ss * d2R_qyyT;
				const Eigen::Matrix3d d2cov_ss_qyz = d2R_qyz * Rcov2R_ss + dR_qy * Rcov2R_ss * dR_qzT + dR_qz * Rcov2R_ss * dR_qyT + Rcov2R_ss * d2R_qyzT;
				const Eigen::Matrix3d d2cov_ss_qzz = d2R_qzz * Rcov2R_ss + 2.0 * dR_qz * Rcov2R_ss * dR_qzT + Rcov2R_ss * d2R_qzzT;

				const Eigen::Matrix3d d2invcov_ss_qxx = -dinvcov_ss_qx * dcov_ss_qx * invcov_ss - invcov_ss * d2cov_ss_qxx * invcov_ss - invcov_ss * dcov_ss_qx * dinvcov_ss_qx;
				const Eigen::Matrix3d d2invcov_ss_qxy = -dinvcov_ss_qy * dcov_ss_qx * invcov_ss - invcov_ss * d2cov_ss_qxy * invcov_ss - invcov_ss * dcov_ss_qx * dinvcov_ss_qy;
				const Eigen::Matrix3d d2invcov_ss_qxz = -dinvcov_ss_qz * dcov_ss_qx * invcov_ss - invcov_ss * d2cov_ss_qxz * invcov_ss - invcov_ss * dcov_ss_qx * dinvcov_ss_qz;
				const Eigen::Matrix3d d2invcov_ss_qyy = -dinvcov_ss_qy * dcov_ss_qy * invcov_ss - invcov_ss * d2cov_ss_qyy * invcov_ss - invcov_ss * dcov_ss_qy * dinvcov_ss_qy;
				const Eigen::Matrix3d d2invcov_ss_qyz = -dinvcov_ss_qz * dcov_ss_qy * invcov_ss - invcov_ss * d2cov_ss_qyz * invcov_ss - invcov_ss * dcov_ss_qy * dinvcov_ss_qz;
				const Eigen::Matrix3d d2invcov_ss_qzz = -dinvcov_ss_qz * dcov_ss_qz * invcov_ss - invcov_ss * d2cov_ss_qzz * invcov_ss - invcov_ss * dcov_ss_qz * dinvcov_ss_qz;

				const Eigen::Vector3d invcov_ss_ddiff_s_tx = invcov_ss * ddiff_s_tx;
				const Eigen::Vector3d invcov_ss_ddiff_s_ty = invcov_ss * ddiff_s_ty;
				const Eigen::Vector3d invcov_ss_ddiff_s_tz = invcov_ss * ddiff_s_tz;
				const Eigen::Vector3d invcov_ss_ddiff_s_qx = invcov_ss * ddiff_s_qx;
				const Eigen::Vector3d invcov_ss_ddiff_s_qy = invcov_ss * ddiff_s_qy;
				const Eigen::Vector3d invcov_ss_ddiff_s_qz = invcov_ss * ddiff_s_qz;

				d2J_pp(0,0) = 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_tx );
				d2J_pp(0,1) = 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_ty );
				d2J_pp(0,2) = 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_tz );
				d2J_pp(0,3) = 2.0 * ddiff_s_tx.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_qx );
				d2J_pp(0,4) = 2.0 * ddiff_s_tx.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_qy );
				d2J_pp(0,5) = 2.0 * ddiff_s_tx.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_qz );

				d2J_pp(1,0) = d2J_pp(0,1);
				d2J_pp(1,1) = 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_ty );
				d2J_pp(1,2) = 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_tz );
				d2J_pp(1,3) = 2.0 * ddiff_s_ty.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_qx );
				d2J_pp(1,4) = 2.0 * ddiff_s_ty.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_qy );
				d2J_pp(1,5) = 2.0 * ddiff_s_ty.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_qz );

				d2J_pp(2,0) = d2J_pp(0,2);
				d2J_pp(2,1) = d2J_pp(1,2);
				d2J_pp(2,2) = 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_tz );
				d2J_pp(2,3) = 2.0 * ddiff_s_tz.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_qx );
				d2J_pp(2,4) = 2.0 * ddiff_s_tz.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_qy );
				d2J_pp(2,5) = 2.0 * ddiff_s_tz.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_qz );

				d2J_pp(3,0) = d2J_pp(0,3);
				d2J_pp(3,1) = d2J_pp(1,3);
				d2J_pp(3,2) = d2J_pp(2,3);
				d2J_pp(3,3) = 2.0 * d2diff_s_qxx.dot( invcov_ss_diff_s ) + 4.0 * ddiff_s_qx.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_qx.dot( invcov_ss_ddiff_s_qx ) + diff_s.dot( d2invcov_ss_qxx * diff_s );
				d2J_pp(3,4) = 2.0 * d2diff_s_qxy.dot( invcov_ss_diff_s ) + 2.0 * ddiff_s_qx.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_qx.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_s_qy.dot( dinvcov_ss_qx_diff_s ) + diff_s.dot( d2invcov_ss_qxy * diff_s );
				d2J_pp(3,5) = 2.0 * d2diff_s_qxz.dot( invcov_ss_diff_s ) + 2.0 * ddiff_s_qx.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_qx.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_s_qz.dot( dinvcov_ss_qx_diff_s ) + diff_s.dot( d2invcov_ss_qxz * diff_s );

				d2J_pp(4,0) = d2J_pp(0,4);
				d2J_pp(4,1) = d2J_pp(1,4);
				d2J_pp(4,2) = d2J_pp(2,4);
				d2J_pp(4,3) = d2J_pp(3,4);
				d2J_pp(4,4) = 2.0 * d2diff_s_qyy.dot( invcov_ss_diff_s ) + 4.0 * ddiff_s_qy.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_qy.dot( invcov_ss_ddiff_s_qy ) + diff_s.dot( d2invcov_ss_qyy * diff_s );
				d2J_pp(4,5) = 2.0 * d2diff_s_qyz.dot( invcov_ss_diff_s ) + 2.0 * ddiff_s_qy.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_qy.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_s_qz.dot( dinvcov_ss_qy_diff_s ) + diff_s.dot( d2invcov_ss_qyz * diff_s );

				d2J_pp(5,0) = d2J_pp(0,5);
				d2J_pp(5,1) = d2J_pp(1,5);
				d2J_pp(5,2) = d2J_pp(2,5);
				d2J_pp(5,3) = d2J_pp(3,5);
				d2J_pp(5,4) = d2J_pp(4,5);
				d2J_pp(5,5) = 2.0 * d2diff_s_qzz.dot( invcov_ss_diff_s ) + 4.0 * ddiff_s_qz.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_qz.dot( invcov_ss_ddiff_s_qz ) + diff_s.dot( d2invcov_ss_qzz * diff_s );


				// further terms: derivative for normalizer of the normal distribution! det(cov) is not independent of q!
				// = dtr( cov^-1 * dcov/dq ) / dq
				// = tr( d( cov^-1 * dcov/dq ) / dq )
				// = tr( dcov^-1/dq * dcov/dq + cov^-1 * d2cov/dqq )
				d2J_pp(0,0) += (dinvcov_ss_qx * dcov_ss_qx + invcov_ss * d2cov_ss_qxx).trace();
				d2J_pp(0,1) += (dinvcov_ss_qy * dcov_ss_qx + invcov_ss * d2cov_ss_qxy).trace();
				d2J_pp(0,2) += (dinvcov_ss_qz * dcov_ss_qx + invcov_ss * d2cov_ss_qxz).trace();
				d2J_pp(1,0) = d2J_pp(0,1);
				d2J_pp(1,1) += (dinvcov_ss_qy * dcov_ss_qy + invcov_ss * d2cov_ss_qyy).trace();
				d2J_pp(1,2) += (dinvcov_ss_qz * dcov_ss_qy + invcov_ss * d2cov_ss_qyz).trace();
				d2J_pp(2,0) = d2J_pp(0,2);
				d2J_pp(2,1) = d2J_pp(1,2);
				d2J_pp(2,2) += (dinvcov_ss_qz * dcov_ss_qz + invcov_ss * d2cov_ss_qzz).trace();


				if( derivZ_ ) {

					// structure: pose along rows; first model coordinates, then scene
					Eigen::Matrix< double, 6, 3 > d2J_pzm, d2J_pzs;
					d2J_pzm(0,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_tx );
					d2J_pzm(0,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_tx );
					d2J_pzm(0,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_tx );
					d2J_pzs(0,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_tx );
					d2J_pzs(0,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_tx );
					d2J_pzs(0,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_tx );
					d2J_pzm(1,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_ty );
					d2J_pzm(1,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_ty );
					d2J_pzm(1,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_ty );
					d2J_pzs(1,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_ty );
					d2J_pzs(1,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_ty );
					d2J_pzs(1,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_ty );
					d2J_pzm(2,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_tz );
					d2J_pzm(2,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_tz );
					d2J_pzm(2,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_tz );
					d2J_pzs(2,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_tz );
					d2J_pzs(2,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_tz );
					d2J_pzs(2,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_tz );
					d2J_pzm(3,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzmx.dot( dinvcov_ss_qx_diff_s );
					d2J_pzm(3,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzmy.dot( dinvcov_ss_qx_diff_s );
					d2J_pzm(3,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzmz.dot( dinvcov_ss_qx_diff_s );
					d2J_pzs(3,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzsx.dot( dinvcov_ss_qx_diff_s ) + 2.0 * d2diff_qx_zsx.dot( invcov_ss_diff_s );
					d2J_pzs(3,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzsy.dot( dinvcov_ss_qx_diff_s ) + 2.0 * d2diff_qx_zsy.dot( invcov_ss_diff_s );
					d2J_pzs(3,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzsz.dot( dinvcov_ss_qx_diff_s ) + 2.0 * d2diff_qx_zsz.dot( invcov_ss_diff_s );
					d2J_pzm(4,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzmx.dot( dinvcov_ss_qy_diff_s );
					d2J_pzm(4,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzmy.dot( dinvcov_ss_qy_diff_s );
					d2J_pzm(4,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzmz.dot( dinvcov_ss_qy_diff_s );
					d2J_pzs(4,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzsx.dot( dinvcov_ss_qy_diff_s ) + 2.0 * d2diff_qy_zsx.dot( invcov_ss_diff_s );
					d2J_pzs(4,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzsy.dot( dinvcov_ss_qy_diff_s ) + 2.0 * d2diff_qy_zsy.dot( invcov_ss_diff_s );
					d2J_pzs(4,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzsz.dot( dinvcov_ss_qy_diff_s ) + 2.0 * d2diff_qy_zsz.dot( invcov_ss_diff_s );
					d2J_pzm(5,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzmx.dot( dinvcov_ss_qz_diff_s );
					d2J_pzm(5,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzmy.dot( dinvcov_ss_qz_diff_s );
					d2J_pzm(5,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzmz.dot( dinvcov_ss_qz_diff_s );
					d2J_pzs(5,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzsx.dot( dinvcov_ss_qz_diff_s ) + 2.0 * d2diff_qz_zsx.dot( invcov_ss_diff_s );
					d2J_pzs(5,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzsy.dot( dinvcov_ss_qz_diff_s ) + 2.0 * d2diff_qz_zsy.dot( invcov_ss_diff_s );
					d2J_pzs(5,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzsz.dot( dinvcov_ss_qz_diff_s ) + 2.0 * d2diff_qz_zsz.dot( invcov_ss_diff_s );


					JSzJ += d2J_pzm * cov1_ss * d2J_pzm.transpose();
					JSzJ += d2J_pzs * cov2_ss * d2J_pzs.transpose();

				}

			}

		}
		else {

			const Eigen::Vector3d ddiff_s_qx = -dR_qx * srcMean;
			const Eigen::Vector3d ddiff_s_qy = -dR_qy * srcMean;
			const Eigen::Vector3d ddiff_s_qz = -dR_qz * srcMean;

			const Eigen::Matrix3d dcov_ss_qx = dR_qx * Rcov2_ssT + Rcov2_ss * dR_qxT;
			const Eigen::Matrix3d dcov_ss_qy = dR_qy * Rcov2_ssT + Rcov2_ss * dR_qyT;
			const Eigen::Matrix3d dcov_ss_qz = dR_qz * Rcov2_ssT + Rcov2_ss * dR_qzT;

			const Eigen::Matrix3d dinvcov_ss_qx = -invcov_ss * dcov_ss_qx * invcov_ss;
			const Eigen::Matrix3d dinvcov_ss_qy = -invcov_ss * dcov_ss_qy * invcov_ss;
			const Eigen::Matrix3d dinvcov_ss_qz = -invcov_ss * dcov_ss_qz * invcov_ss;

			const Eigen::Vector3d dinvcov_ss_qx_diff_s = dinvcov_ss_qx * diff_s;
			const Eigen::Vector3d dinvcov_ss_qy_diff_s = dinvcov_ss_qy * diff_s;
			const Eigen::Vector3d dinvcov_ss_qz_diff_s = dinvcov_ss_qz * diff_s;


			de_tx = 2.0 * ddiff_s_tx.dot(invcov_ss_diff_s);
			de_ty = 2.0 * ddiff_s_ty.dot(invcov_ss_diff_s);
			de_tz = 2.0 * ddiff_s_tz.dot(invcov_ss_diff_s);
			de_qx = 2.0 * ddiff_s_qx.dot(invcov_ss_diff_s) + diff_s.dot(dinvcov_ss_qx_diff_s);
			de_qy = 2.0 * ddiff_s_qy.dot(invcov_ss_diff_s) + diff_s.dot(dinvcov_ss_qy_diff_s);
			de_qz = 2.0 * ddiff_s_qz.dot(invcov_ss_diff_s) + diff_s.dot(dinvcov_ss_qz_diff_s);

			// second term: derivative for normalizer of the normal distribution! det(cov) is not independent of q!
			// -log( (2pi)^-(3/2) (det(cov))^(-1/2) )
			// = - log( (2pi)^-(3/2) ) - log( (det(cov))^(-1/2) )
			// = const. - (-0.5) * log( det(cov) )
			// = 0.5 * log( det(cov) ) => 0.5 factor can be left out also for the exp part...
			// d(log(det(cov)))/dq = 1/det(cov) * det(cov) * tr( cov^-1 * dcov/dq )
			// = tr( cov^-1 * dcov/dq )
			de_qx += (invcov_ss * dcov_ss_qx).trace();
			de_qy += (invcov_ss * dcov_ss_qy).trace();
			de_qz += (invcov_ss * dcov_ss_qz).trace();


			if( deriv2_ ) {

				const Eigen::Vector3d d2diff_s_qxx = -d2R_qxx * srcMean;
				const Eigen::Vector3d d2diff_s_qxy = -d2R_qxy * srcMean;
				const Eigen::Vector3d d2diff_s_qxz = -d2R_qxz * srcMean;
				const Eigen::Vector3d d2diff_s_qyy = -d2R_qyy * srcMean;
				const Eigen::Vector3d d2diff_s_qyz = -d2R_qyz * srcMean;
				const Eigen::Vector3d d2diff_s_qzz = -d2R_qzz * srcMean;

				const Eigen::Matrix3d d2cov_ss_qxx = d2R_qxx * Rcov2_ssT + 2.0 * dR_qx * cov2_ss * dR_qxT + Rcov2_ss * d2R_qxxT;
				const Eigen::Matrix3d d2cov_ss_qxy = d2R_qxy * Rcov2_ssT + dR_qx * cov2_ss * dR_qyT + dR_qy * cov2_ss * dR_qxT + Rcov2_ss * d2R_qxyT;
				const Eigen::Matrix3d d2cov_ss_qxz = d2R_qxz * Rcov2_ssT + dR_qx * cov2_ss * dR_qzT + dR_qz * cov2_ss * dR_qxT + Rcov2_ss * d2R_qxzT;
				const Eigen::Matrix3d d2cov_ss_qyy = d2R_qyy * Rcov2_ssT + 2.0 * dR_qy * cov2_ss * dR_qyT + Rcov2_ss * d2R_qyyT;
				const Eigen::Matrix3d d2cov_ss_qyz = d2R_qyz * Rcov2_ssT + dR_qy * cov2_ss * dR_qzT + dR_qz * cov2_ss * dR_qyT + Rcov2_ss * d2R_qyzT;
				const Eigen::Matrix3d d2cov_ss_qzz = d2R_qzz * Rcov2_ssT + 2.0 * dR_qz * cov2_ss * dR_qzT + Rcov2_ss * d2R_qzzT;

				const Eigen::Matrix3d d2invcov_ss_qxx = -dinvcov_ss_qx * dcov_ss_qx * invcov_ss - invcov_ss * d2cov_ss_qxx * invcov_ss - invcov_ss * dcov_ss_qx * dinvcov_ss_qx;
				const Eigen::Matrix3d d2invcov_ss_qxy = -dinvcov_ss_qy * dcov_ss_qx * invcov_ss - invcov_ss * d2cov_ss_qxy * invcov_ss - invcov_ss * dcov_ss_qx * dinvcov_ss_qy;
				const Eigen::Matrix3d d2invcov_ss_qxz = -dinvcov_ss_qz * dcov_ss_qx * invcov_ss - invcov_ss * d2cov_ss_qxz * invcov_ss - invcov_ss * dcov_ss_qx * dinvcov_ss_qz;
				const Eigen::Matrix3d d2invcov_ss_qyy = -dinvcov_ss_qy * dcov_ss_qy * invcov_ss - invcov_ss * d2cov_ss_qyy * invcov_ss - invcov_ss * dcov_ss_qy * dinvcov_ss_qy;
				const Eigen::Matrix3d d2invcov_ss_qyz = -dinvcov_ss_qz * dcov_ss_qy * invcov_ss - invcov_ss * d2cov_ss_qyz * invcov_ss - invcov_ss * dcov_ss_qy * dinvcov_ss_qz;
				const Eigen::Matrix3d d2invcov_ss_qzz = -dinvcov_ss_qz * dcov_ss_qz * invcov_ss - invcov_ss * d2cov_ss_qzz * invcov_ss - invcov_ss * dcov_ss_qz * dinvcov_ss_qz;

				const Eigen::Vector3d invcov_ss_ddiff_s_tx = invcov_ss * ddiff_s_tx;
				const Eigen::Vector3d invcov_ss_ddiff_s_ty = invcov_ss * ddiff_s_ty;
				const Eigen::Vector3d invcov_ss_ddiff_s_tz = invcov_ss * ddiff_s_tz;
				const Eigen::Vector3d invcov_ss_ddiff_s_qx = invcov_ss * ddiff_s_qx;
				const Eigen::Vector3d invcov_ss_ddiff_s_qy = invcov_ss * ddiff_s_qy;
				const Eigen::Vector3d invcov_ss_ddiff_s_qz = invcov_ss * ddiff_s_qz;

				d2J_pp(0,0) = 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_tx );
				d2J_pp(0,1) = 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_ty );
				d2J_pp(0,2) = 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_tz );
				d2J_pp(0,3) = 2.0 * ddiff_s_tx.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_qx );
				d2J_pp(0,4) = 2.0 * ddiff_s_tx.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_qy );
				d2J_pp(0,5) = 2.0 * ddiff_s_tx.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_qz );

				d2J_pp(1,0) = d2J_pp(0,1);
				d2J_pp(1,1) = 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_ty );
				d2J_pp(1,2) = 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_tz );
				d2J_pp(1,3) = 2.0 * ddiff_s_ty.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_qx );
				d2J_pp(1,4) = 2.0 * ddiff_s_ty.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_qy );
				d2J_pp(1,5) = 2.0 * ddiff_s_ty.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_qz );

				d2J_pp(2,0) = d2J_pp(0,2);
				d2J_pp(2,1) = d2J_pp(1,2);
				d2J_pp(2,2) = 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_tz );
				d2J_pp(2,3) = 2.0 * ddiff_s_tz.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_qx );
				d2J_pp(2,4) = 2.0 * ddiff_s_tz.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_qy );
				d2J_pp(2,5) = 2.0 * ddiff_s_tz.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_qz );

				d2J_pp(3,0) = d2J_pp(0,3);
				d2J_pp(3,1) = d2J_pp(1,3);
				d2J_pp(3,2) = d2J_pp(2,3);
				d2J_pp(3,3) = 2.0 * d2diff_s_qxx.dot( invcov_ss_diff_s ) + 4.0 * ddiff_s_qx.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_qx.dot( invcov_ss_ddiff_s_qx ) + diff_s.dot( d2invcov_ss_qxx * diff_s );
				d2J_pp(3,4) = 2.0 * d2diff_s_qxy.dot( invcov_ss_diff_s ) + 2.0 * ddiff_s_qx.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_qx.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_s_qy.dot( dinvcov_ss_qx_diff_s ) + diff_s.dot( d2invcov_ss_qxy * diff_s );
				d2J_pp(3,5) = 2.0 * d2diff_s_qxz.dot( invcov_ss_diff_s ) + 2.0 * ddiff_s_qx.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_qx.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_s_qz.dot( dinvcov_ss_qx_diff_s ) + diff_s.dot( d2invcov_ss_qxz * diff_s );

				d2J_pp(4,0) = d2J_pp(0,4);
				d2J_pp(4,1) = d2J_pp(1,4);
				d2J_pp(4,2) = d2J_pp(2,4);
				d2J_pp(4,3) = d2J_pp(3,4);
				d2J_pp(4,4) = 2.0 * d2diff_s_qyy.dot( invcov_ss_diff_s ) + 4.0 * ddiff_s_qy.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_qy.dot( invcov_ss_ddiff_s_qy ) + diff_s.dot( d2invcov_ss_qyy * diff_s );
				d2J_pp(4,5) = 2.0 * d2diff_s_qyz.dot( invcov_ss_diff_s ) + 2.0 * ddiff_s_qy.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_qy.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_s_qz.dot( dinvcov_ss_qy_diff_s ) + diff_s.dot( d2invcov_ss_qyz * diff_s );

				d2J_pp(5,0) = d2J_pp(0,5);
				d2J_pp(5,1) = d2J_pp(1,5);
				d2J_pp(5,2) = d2J_pp(2,5);
				d2J_pp(5,3) = d2J_pp(3,5);
				d2J_pp(5,4) = d2J_pp(4,5);
				d2J_pp(5,5) = 2.0 * d2diff_s_qzz.dot( invcov_ss_diff_s ) + 4.0 * ddiff_s_qz.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_qz.dot( invcov_ss_ddiff_s_qz ) + diff_s.dot( d2invcov_ss_qzz * diff_s );


				// further terms: derivative for normalizer of the normal distribution! det(cov) is not independent of q!
				// = dtr( cov^-1 * dcov/dq ) / dq
				// = tr( d( cov^-1 * dcov/dq ) / dq )
				// = tr( dcov^-1/dq * dcov/dq + cov^-1 * d2cov/dqq )
				d2J_pp(0,0) += (dinvcov_ss_qx * dcov_ss_qx + invcov_ss * d2cov_ss_qxx).trace();
				d2J_pp(0,1) += (dinvcov_ss_qy * dcov_ss_qx + invcov_ss * d2cov_ss_qxy).trace();
				d2J_pp(0,2) += (dinvcov_ss_qz * dcov_ss_qx + invcov_ss * d2cov_ss_qxz).trace();
				d2J_pp(1,0) = d2J_pp(0,1);
				d2J_pp(1,1) += (dinvcov_ss_qy * dcov_ss_qy + invcov_ss * d2cov_ss_qyy).trace();
				d2J_pp(1,2) += (dinvcov_ss_qz * dcov_ss_qy + invcov_ss * d2cov_ss_qyz).trace();
				d2J_pp(2,0) = d2J_pp(0,2);
				d2J_pp(2,1) = d2J_pp(1,2);
				d2J_pp(2,2) += (dinvcov_ss_qz * dcov_ss_qz + invcov_ss * d2cov_ss_qzz).trace();


				if( derivZ_ ) {

					// structure: pose along rows; first model coordinates, then scene
					Eigen::Matrix< double, 6, 3 > d2J_pzm, d2J_pzs;
					d2J_pzm(0,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_tx );
					d2J_pzm(0,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_tx );
					d2J_pzm(0,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_tx );
					d2J_pzs(0,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_tx );
					d2J_pzs(0,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_tx );
					d2J_pzs(0,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_tx );
					d2J_pzm(1,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_ty );
					d2J_pzm(1,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_ty );
					d2J_pzm(1,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_ty );
					d2J_pzs(1,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_ty );
					d2J_pzs(1,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_ty );
					d2J_pzs(1,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_ty );
					d2J_pzm(2,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_tz );
					d2J_pzm(2,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_tz );
					d2J_pzm(2,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_tz );
					d2J_pzs(2,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_tz );
					d2J_pzs(2,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_tz );
					d2J_pzs(2,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_tz );
					d2J_pzm(3,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzmx.dot( dinvcov_ss_qx_diff_s );
					d2J_pzm(3,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzmy.dot( dinvcov_ss_qx_diff_s );
					d2J_pzm(3,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzmz.dot( dinvcov_ss_qx_diff_s );
					d2J_pzs(3,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzsx.dot( dinvcov_ss_qx_diff_s ) + 2.0 * d2diff_qx_zsx.dot( invcov_ss_diff_s );
					d2J_pzs(3,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzsy.dot( dinvcov_ss_qx_diff_s ) + 2.0 * d2diff_qx_zsy.dot( invcov_ss_diff_s );
					d2J_pzs(3,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzsz.dot( dinvcov_ss_qx_diff_s ) + 2.0 * d2diff_qx_zsz.dot( invcov_ss_diff_s );
					d2J_pzm(4,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzmx.dot( dinvcov_ss_qy_diff_s );
					d2J_pzm(4,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzmy.dot( dinvcov_ss_qy_diff_s );
					d2J_pzm(4,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzmz.dot( dinvcov_ss_qy_diff_s );
					d2J_pzs(4,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzsx.dot( dinvcov_ss_qy_diff_s ) + 2.0 * d2diff_qy_zsx.dot( invcov_ss_diff_s );
					d2J_pzs(4,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzsy.dot( dinvcov_ss_qy_diff_s ) + 2.0 * d2diff_qy_zsy.dot( invcov_ss_diff_s );
					d2J_pzs(4,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzsz.dot( dinvcov_ss_qy_diff_s ) + 2.0 * d2diff_qy_zsz.dot( invcov_ss_diff_s );
					d2J_pzm(5,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzmx.dot( dinvcov_ss_qz_diff_s );
					d2J_pzm(5,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzmy.dot( dinvcov_ss_qz_diff_s );
					d2J_pzm(5,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzmz.dot( dinvcov_ss_qz_diff_s );
					d2J_pzs(5,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzsx.dot( dinvcov_ss_qz_diff_s ) + 2.0 * d2diff_qz_zsx.dot( invcov_ss_diff_s );
					d2J_pzs(5,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzsy.dot( dinvcov_ss_qz_diff_s ) + 2.0 * d2diff_qz_zsy.dot( invcov_ss_diff_s );
					d2J_pzs(5,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzsz.dot( dinvcov_ss_qz_diff_s ) + 2.0 * d2diff_qz_zsz.dot( invcov_ss_diff_s );


					JSzJ += d2J_pzm * cov1_ss * d2J_pzm.transpose();
					JSzJ += d2J_pzs * cov2_ss * d2J_pzs.transpose();

				}

			}

		}


		assoc.df_dx(0) = de_tx;
		assoc.df_dx(1) = de_ty;
		assoc.df_dx(2) = de_tz;
		assoc.df_dx(3) = de_qx;
		assoc.df_dx(4) = de_qy;
		assoc.df_dx(5) = de_qz;

		if( deriv2_ ) {
			assoc.d2f = d2J_pp;

			if( derivZ_ )
				assoc.JSzJ = JSzJ;
		}

		assoc.error = error;
		assoc.weight = weight;
		assoc.match = 1;

		assert( !isnan(error) );




	}


	MultiResolutionColorSurfelRegistration::Params params_;

	double tx, ty, tz, qx, qy, qz, qw;
	Eigen::Matrix4d currentTransform;
	Eigen::Vector3d ddiff_s_tx, ddiff_s_ty, ddiff_s_tz;
	Eigen::Matrix3d dR_qx, dR_qy, dR_qz;
	Eigen::Matrix3d dR_qxT, dR_qyT, dR_qzT;
	Eigen::Vector3d dt_tx, dt_ty, dt_tz;
//	Eigen::Matrix3d cov_cc_add;
	Eigen::Matrix3d currentRotation;
	Eigen::Matrix3d currentRotationT;
	Eigen::Vector3d currentTranslation;

	// 2nd order derivatives
	Eigen::Matrix3d d2R_qxx, d2R_qxy, d2R_qxz, d2R_qyy, d2R_qyz, d2R_qzz;
	Eigen::Matrix3d d2R_qxxT, d2R_qxyT, d2R_qxzT, d2R_qyyT, d2R_qyzT, d2R_qzzT;

	// 1st and 2nd order derivatives on Z
	Eigen::Vector3d ddiff_dzsx, ddiff_dzsy, ddiff_dzsz;
	Eigen::Vector3d ddiff_dzmx, ddiff_dzmy, ddiff_dzmz;
	Eigen::Vector3d d2diff_qx_zsx, d2diff_qx_zsy, d2diff_qx_zsz;
	Eigen::Vector3d d2diff_qy_zsx, d2diff_qy_zsy, d2diff_qy_zsz;
	Eigen::Vector3d d2diff_qz_zsx, d2diff_qz_zsy, d2diff_qz_zsz;


	bool relativeDerivatives_;
	bool deriv2_, derivZ_;
	bool interpolate_neighbors_;

	MultiResolutionColorSurfelRegistration::SurfelAssociationList* assocList_;

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};


class GradientFunctorLM {
public:


	GradientFunctorLM( MultiResolutionColorSurfelRegistration::SurfelAssociationList* assocList, const MultiResolutionColorSurfelRegistration::Params& params, double tx, double ty, double tz, double qx, double qy, double qz, double qw, bool derivs, bool derivZ = false, bool interpolate_neighbors = false ) {

		derivs_ = derivs;
		derivZ_ = derivZ;

		interpolate_neighbors_ = interpolate_neighbors;

		assocList_ = assocList;

		params_ = params;

		currentTransform.setIdentity();
		currentTransform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
		currentTransform(0,3) = tx;
		currentTransform(1,3) = ty;
		currentTransform(2,3) = tz;

		currentRotation = Eigen::Matrix3d( currentTransform.block<3,3>(0,0) );
		currentRotationT = currentRotation.transpose();
		currentTranslation = Eigen::Vector3d( currentTransform.block<3,1>(0,3) );

		if( derivs ) {

			const double inv_qw = 1.0 / qw;

			// build up derivatives of rotation and translation for the transformation variables
			dt_tx(0) = 1.f; dt_tx(1) = 0.f; dt_tx(2) = 0.f;
			dt_ty(0) = 0.f; dt_ty(1) = 1.f; dt_ty(2) = 0.f;
			dt_tz(0) = 0.f; dt_tz(1) = 0.f; dt_tz(2) = 1.f;

			if( !derivZ ) {
				dR_qx.setZero();
				dR_qx(1,2) = -2;
				dR_qx(2,1) = 2;

				dR_qy.setZero();
				dR_qy(0,2) = 2;
				dR_qy(2,0) = -2;

				dR_qz.setZero();
				dR_qz(0,1) = -2;
				dR_qz(1,0) = 2;
			}
			else {

                // matrix(
                //  [ 0,
                //    2*((qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qy),
                //    2*(qz-(qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)) ],
                //  [ 2*(qy-(qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)),
                //    -4*qx,
                //    2*(qx^2/sqrt(-qz^2-qy^2-qx^2+1)-sqrt(-qz^2-qy^2-qx^2+1)) ],
                //  [ 2*((qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)+qz),
                //    2*(sqrt(-qz^2-qy^2-qx^2+1)-qx^2/sqrt(-qz^2-qy^2-qx^2+1)),
                //    -4*qx ]
                // )
                dR_qx(0,0) = 0.0;
                dR_qx(0,1) = 2.0*((qx*qz)*inv_qw+qy);
                dR_qx(0,2) = 2.0*(qz-(qx*qy)*inv_qw);
                dR_qx(1,0) = 2.0*(qy-(qx*qz)*inv_qw);
                dR_qx(1,1) = -4.0*qx;
                dR_qx(1,2) = 2.0*(qx*qx*inv_qw-qw);
                dR_qx(2,0) = 2.0*((qx*qy)*inv_qw+qz);
                dR_qx(2,1) = 2.0*(qw-qx*qx*inv_qw);
                dR_qx(2,2) = -4.0*qx;

                // matrix(
                //  [ -4*qy,
                //    2*((qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qx),
                //    2*(sqrt(-qz^2-qy^2-qx^2+1)-qy^2/sqrt(-qz^2-qy^2-qx^2+1)) ],
                //  [ 2*(qx-(qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)),
                //    0,
                //    2*((qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)+qz) ],
                //  [ 2*(qy^2/sqrt(-qz^2-qy^2-qx^2+1)-sqrt(-qz^2-qy^2-qx^2+1)),
                //    2*(qz-(qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)),
                //    -4*qy ]
                // )

                dR_qy(0,0) = -4.0*qy;
                dR_qy(0,1) = 2.0*((qy*qz)*inv_qw+qx);
                dR_qy(0,2) = 2.0*(qw-qy*qy*inv_qw);
                dR_qy(1,0) = 2.0*(qx-(qy*qz)*inv_qw);
                dR_qy(1,1) = 0.0;
                dR_qy(1,2) = 2.0*((qx*qy)*inv_qw+qz);
                dR_qy(2,0) = 2.0*(qy*qy*inv_qw-qw);
                dR_qy(2,1) = 2.0*(qz-(qx*qy)*inv_qw);
                dR_qy(2,2) = -4.0*qy;


                // matrix(
                //  [ -4*qz,
                //    2*(qz^2/sqrt(-qz^2-qy^2-qx^2+1)-sqrt(-qz^2-qy^2-qx^2+1)),
                //    2*(qx-(qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)) ],
                //  [ 2*(sqrt(-qz^2-qy^2-qx^2+1)-qz^2/sqrt(-qz^2-qy^2-qx^2+1)),
                //    -4*qz,
                //    2*((qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qy) ],
                //  [ 2*((qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qx),
                //    2*(qy-(qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)),
                //    0 ]
                // )
                dR_qz(0,0) = -4.0*qz;
                dR_qz(0,1) = 2.0*(qz*qz*inv_qw-qw);
                dR_qz(0,2) = 2.0*(qx-(qy*qz)*inv_qw);
                dR_qz(1,0) = 2.0*(qw-qz*qz*inv_qw);
                dR_qz(1,1) = -4.0*qz;
                dR_qz(1,2) = 2.0*((qx*qz)*inv_qw+qy);
                dR_qz(2,0) = 2.0*((qy*qz)*inv_qw+qx);
                dR_qz(2,1) = 2.0*(qy-(qx*qz)*inv_qw);
                dR_qz(2,2) = 0.0;


                // needed for the derivatives for the measurements

                ddiff_dzmx = Eigen::Vector3d( 1.0, 0.0, 0.0 );
                ddiff_dzmy = Eigen::Vector3d( 0.0, 1.0, 0.0 );
                ddiff_dzmz = Eigen::Vector3d( 0.0, 0.0, 1.0 );

                ddiff_dzsx = -currentRotation * Eigen::Vector3d( 1.0, 0.0, 0.0 );
                ddiff_dzsy = -currentRotation * Eigen::Vector3d( 0.0, 1.0, 0.0 );
                ddiff_dzsz = -currentRotation * Eigen::Vector3d( 0.0, 0.0, 1.0 );

                d2diff_qx_zsx = -dR_qx * Eigen::Vector3d( 1.0, 0.0, 0.0 );
                d2diff_qx_zsy = -dR_qx * Eigen::Vector3d( 0.0, 1.0, 0.0 );
                d2diff_qx_zsz = -dR_qx * Eigen::Vector3d( 0.0, 0.0, 1.0 );
                d2diff_qy_zsx = -dR_qy * Eigen::Vector3d( 1.0, 0.0, 0.0 );
                d2diff_qy_zsy = -dR_qy * Eigen::Vector3d( 0.0, 1.0, 0.0 );
                d2diff_qy_zsz = -dR_qy * Eigen::Vector3d( 0.0, 0.0, 1.0 );
                d2diff_qz_zsx = -dR_qz * Eigen::Vector3d( 1.0, 0.0, 0.0 );
                d2diff_qz_zsy = -dR_qz * Eigen::Vector3d( 0.0, 1.0, 0.0 );
                d2diff_qz_zsz = -dR_qz * Eigen::Vector3d( 0.0, 0.0, 1.0 );


			}

		}

	}

	~GradientFunctorLM() {}


	void operator()( const tbb::blocked_range<size_t>& r ) const {
		for( size_t i=r.begin(); i!=r.end(); ++i )
			(*this)((*assocList_)[i]);
	}



	void operator()( MultiResolutionColorSurfelRegistration::SurfelAssociation& assoc ) const {


		if( assoc.match == 0 || !assoc.src_->applyUpdate_ || !assoc.dst_->applyUpdate_ ) {
			assoc.match = 0;
			return;
		}


		const float processResolution = assoc.n_src_->resolution();

		Eigen::Matrix3d cov_ss_add;
		cov_ss_add.setZero();
		if( params_.add_smooth_pos_covariance_ ) {
			cov_ss_add.setIdentity();
			cov_ss_add *= params_.smooth_surface_cov_factor_ * processResolution*processResolution;
		}

		Eigen::Matrix3d cov1_ss;
		Eigen::Matrix3d cov2_ss = assoc.src_->cov_.block<3,3>(0,0) + cov_ss_add;

		Eigen::Vector3d dstMean;
		Eigen::Vector3d srcMean = assoc.src_->mean_.block<3,1>(0,0);

		Eigen::Vector4d pos;
		pos.block<3,1>(0,0) = srcMean;
		pos(3,0) = 1.f;

		const Eigen::Vector4d pos_src = currentTransform * pos;

		bool in_interpolation_range = false;

		if( interpolate_neighbors_ ) {

			// use trilinear interpolation to handle discretization effects
			// => associate with neighbors and weight correspondences
			// only makes sense when match is within resolution distance to the node center
			const float resolution = processResolution;
			Eigen::Vector3d centerDiff = assoc.n_dst_->getCenterPosition().block<3,1>(0,0).cast<double>() - pos_src.block<3,1>(0,0);
			if( resolution - fabsf(centerDiff(0)) > 0  && resolution - fabsf(centerDiff(1)) > 0  && resolution - fabsf(centerDiff(2)) > 0 ) {

				in_interpolation_range = true;

				// associate with neighbors for which distance to the node center is smaller than resolution

				dstMean.setZero();
				cov1_ss.setZero();

				double sumWeight = 0.f;
				double sumWeight2 = 0.f;

				for( int s = 0; s < 27; s++ ) {

					spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_dst_n = assoc.n_dst_->neighbors_[s];

					if(!n_dst_n)
						continue;

					MultiResolutionColorSurfelMap::Surfel* dst_n = &n_dst_n->value_.surfels_[assoc.dst_idx_];
					if( dst_n->num_points_ < MIN_SURFEL_POINTS )
						continue;

					Eigen::Vector3d centerDiff_n = n_dst_n->getCenterPosition().block<3,1>(0,0).cast<double>() - pos_src.block<3,1>(0,0);
					const double dx = resolution - fabsf(centerDiff_n(0));
					const double dy = resolution - fabsf(centerDiff_n(1));
					const double dz = resolution - fabsf(centerDiff_n(2));

					if( dx > 0 && dy > 0 && dz > 0 ) {

						const double weight = dx*dy*dz;

						dstMean += weight * dst_n->mean_.block<3,1>(0,0);
						cov1_ss += weight*weight * (dst_n->cov_.block<3,3>(0,0));

						sumWeight += weight;
						sumWeight2 += weight*weight;

					}


				}

				// numerically stable?
				if( sumWeight > resolution* 1e-6 ) {
					dstMean /= sumWeight;
					cov1_ss /= sumWeight2;

				}
				else
					in_interpolation_range = false;

				cov1_ss += cov_ss_add;


			}

		}

		if( !interpolate_neighbors_ || !in_interpolation_range ) {

			dstMean = assoc.dst_->mean_.block<3,1>(0,0);
			cov1_ss = assoc.dst_->cov_.block<3,3>(0,0) + cov_ss_add;

		}


		// has only marginal (positive!) effect on visual odometry result
		// makes tracking more robust (when only few surfels available)
		cov1_ss *= INTERPOLATION_COV_FACTOR;
		cov2_ss *= INTERPOLATION_COV_FACTOR;


////		const float processResolution = assoc.n_src_->resolution();
////
////		Eigen::Matrix3d cov_ss_add;
////		cov_ss_add.setZero();
////		if( params_.add_smooth_pos_covariance_ ) {
////			cov_ss_add.setIdentity();
////			cov_ss_add *= params_.smooth_surface_cov_factor_ * processResolution*processResolution;
////		}
//
//		const Eigen::Matrix3d cov1_ss = assoc.dst_->cov_.block<3,3>(0,0);// + cov_ss_add;
//		const Eigen::Matrix3d cov2_ss = assoc.src_->cov_.block<3,3>(0,0);// + cov_ss_add;
//
//		const Eigen::Vector3d dstMean = assoc.dst_->mean_.block<3,1>(0,0);
//		const Eigen::Vector3d srcMean = assoc.src_->mean_.block<3,1>(0,0);



		const Eigen::Vector3d p_s = pos_src.block<3,1>(0,0);
		const Eigen::Vector3d diff_s = dstMean - p_s;

		const Eigen::Matrix3d cov_ss = INTERPOLATION_COV_FACTOR * (cov1_ss + currentRotation * cov2_ss * currentRotationT);
		const Eigen::Matrix3d invcov_ss = cov_ss.inverse();

		assoc.error = diff_s.dot(invcov_ss * diff_s);

		assoc.z = dstMean;
		assoc.h = p_s;

		if( derivs_ ) {

			assoc.dh_dx.block<3,1>(0,0) = dt_tx;
			assoc.dh_dx.block<3,1>(0,1) = dt_ty;
			assoc.dh_dx.block<3,1>(0,2) = dt_tz;
//			assoc.df_dx.block<3,1>(0,3) = dR_qx * srcMean;
//			assoc.df_dx.block<3,1>(0,4) = dR_qy * srcMean;
//			assoc.df_dx.block<3,1>(0,5) = dR_qz * srcMean;
			assoc.dh_dx.block<3,1>(0,3) = dR_qx * pos_src.block<3,1>(0,0);
			assoc.dh_dx.block<3,1>(0,4) = dR_qy * pos_src.block<3,1>(0,0);
			assoc.dh_dx.block<3,1>(0,5) = dR_qz * pos_src.block<3,1>(0,0);

			assoc.W = invcov_ss;

            if( derivZ_ ) {

                    // ddiff_s_X = df_dX.transpose * (z-f)
                    // ddiff_dzmx: simple
                    const Eigen::Vector3d invcov_ss_diff_s = assoc.W * diff_s;
                    const Eigen::Vector3d invcov_ss_ddiff_s_tx = assoc.W * dt_tx;
                    const Eigen::Vector3d invcov_ss_ddiff_s_ty = assoc.W * dt_ty;
                    const Eigen::Vector3d invcov_ss_ddiff_s_tz = assoc.W * dt_tz;
                    const Eigen::Vector3d invcov_ss_ddiff_s_qx = assoc.W * assoc.df_dx.block<3,1>(0,3);
                    const Eigen::Vector3d invcov_ss_ddiff_s_qy = assoc.W * assoc.df_dx.block<3,1>(0,4);
                    const Eigen::Vector3d invcov_ss_ddiff_s_qz = assoc.W * assoc.df_dx.block<3,1>(0,5);


                    // structure: pose along rows; first model coordinates, then scene
                    Eigen::Matrix< double, 6, 3 > d2J_pzm, d2J_pzs;
                    d2J_pzm(0,0) = ddiff_dzmx.dot( invcov_ss_ddiff_s_tx );
                    d2J_pzm(0,1) = ddiff_dzmy.dot( invcov_ss_ddiff_s_tx );
                    d2J_pzm(0,2) = ddiff_dzmz.dot( invcov_ss_ddiff_s_tx );
                    d2J_pzs(0,0) = ddiff_dzsx.dot( invcov_ss_ddiff_s_tx );
                    d2J_pzs(0,1) = ddiff_dzsy.dot( invcov_ss_ddiff_s_tx );
                    d2J_pzs(0,2) = ddiff_dzsz.dot( invcov_ss_ddiff_s_tx );
                    d2J_pzm(1,0) = ddiff_dzmx.dot( invcov_ss_ddiff_s_ty );
                    d2J_pzm(1,1) = ddiff_dzmy.dot( invcov_ss_ddiff_s_ty );
                    d2J_pzm(1,2) = ddiff_dzmz.dot( invcov_ss_ddiff_s_ty );
                    d2J_pzs(1,0) = ddiff_dzsx.dot( invcov_ss_ddiff_s_ty );
                    d2J_pzs(1,1) = ddiff_dzsy.dot( invcov_ss_ddiff_s_ty );
                    d2J_pzs(1,2) = ddiff_dzsz.dot( invcov_ss_ddiff_s_ty );
                    d2J_pzm(2,0) = ddiff_dzmx.dot( invcov_ss_ddiff_s_tz );
                    d2J_pzm(2,1) = ddiff_dzmy.dot( invcov_ss_ddiff_s_tz );
                    d2J_pzm(2,2) = ddiff_dzmz.dot( invcov_ss_ddiff_s_tz );
                    d2J_pzs(2,0) = ddiff_dzsx.dot( invcov_ss_ddiff_s_tz );
                    d2J_pzs(2,1) = ddiff_dzsy.dot( invcov_ss_ddiff_s_tz );
                    d2J_pzs(2,2) = ddiff_dzsz.dot( invcov_ss_ddiff_s_tz );
                    d2J_pzm(3,0) = ddiff_dzmx.dot( invcov_ss_ddiff_s_qx );
                    d2J_pzm(3,1) = ddiff_dzmy.dot( invcov_ss_ddiff_s_qx );
                    d2J_pzm(3,2) = ddiff_dzmz.dot( invcov_ss_ddiff_s_qx );
                    d2J_pzs(3,0) = ddiff_dzsx.dot( invcov_ss_ddiff_s_qx ) + d2diff_qx_zsx.dot( invcov_ss_diff_s );
                    d2J_pzs(3,1) = ddiff_dzsy.dot( invcov_ss_ddiff_s_qx ) + d2diff_qx_zsy.dot( invcov_ss_diff_s );
                    d2J_pzs(3,2) = ddiff_dzsz.dot( invcov_ss_ddiff_s_qx ) + d2diff_qx_zsz.dot( invcov_ss_diff_s );
                    d2J_pzm(4,0) = ddiff_dzmx.dot( invcov_ss_ddiff_s_qy );
                    d2J_pzm(4,1) = ddiff_dzmy.dot( invcov_ss_ddiff_s_qy );
                    d2J_pzm(4,2) = ddiff_dzmz.dot( invcov_ss_ddiff_s_qy );
                    d2J_pzs(4,0) = ddiff_dzsx.dot( invcov_ss_ddiff_s_qy ) + d2diff_qy_zsx.dot( invcov_ss_diff_s );
                    d2J_pzs(4,1) = ddiff_dzsy.dot( invcov_ss_ddiff_s_qy ) + d2diff_qy_zsy.dot( invcov_ss_diff_s );
                    d2J_pzs(4,2) = ddiff_dzsz.dot( invcov_ss_ddiff_s_qy ) + d2diff_qy_zsz.dot( invcov_ss_diff_s );
                    d2J_pzm(5,0) = ddiff_dzmx.dot( invcov_ss_ddiff_s_qz );
                    d2J_pzm(5,1) = ddiff_dzmy.dot( invcov_ss_ddiff_s_qz );
                    d2J_pzm(5,2) = ddiff_dzmz.dot( invcov_ss_ddiff_s_qz );
                    d2J_pzs(5,0) = ddiff_dzsx.dot( invcov_ss_ddiff_s_qz ) + d2diff_qz_zsx.dot( invcov_ss_diff_s );
                    d2J_pzs(5,1) = ddiff_dzsy.dot( invcov_ss_ddiff_s_qz ) + d2diff_qz_zsy.dot( invcov_ss_diff_s );
                    d2J_pzs(5,2) = ddiff_dzsz.dot( invcov_ss_ddiff_s_qz ) + d2diff_qz_zsz.dot( invcov_ss_diff_s );

                    assoc.JSzJ.setZero();
                    assoc.JSzJ += d2J_pzm * cov1_ss * d2J_pzm.transpose();
                    assoc.JSzJ += d2J_pzs * cov2_ss * d2J_pzs.transpose();

            }


		}


		assoc.match = 1;

//		assert( !isnan(assoc.error) );


	}



	Eigen::Matrix4d currentTransform;

	Eigen::Vector3d currentTranslation;
	Eigen::Vector3d dt_tx, dt_ty, dt_tz;

	Eigen::Matrix3d currentRotation, currentRotationT;
	Eigen::Matrix3d dR_qx, dR_qy, dR_qz;

    // 1st and 2nd order derivatives on Z
    Eigen::Vector3d ddiff_dzsx, ddiff_dzsy, ddiff_dzsz;
    Eigen::Vector3d ddiff_dzmx, ddiff_dzmy, ddiff_dzmz;
    Eigen::Vector3d d2diff_qx_zsx, d2diff_qx_zsy, d2diff_qx_zsz;
    Eigen::Vector3d d2diff_qy_zsx, d2diff_qy_zsy, d2diff_qy_zsz;
    Eigen::Vector3d d2diff_qz_zsx, d2diff_qz_zsy, d2diff_qz_zsz;


	MultiResolutionColorSurfelRegistration::SurfelAssociationList* assocList_;

	MultiResolutionColorSurfelRegistration::Params params_;

	bool derivs_, derivZ_, interpolate_neighbors_;

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};


class GradientFunctorPointFeature {
public:

	// dh/dx
	inline Eigen::Matrix<double, 3, 6> dh_dx(const Eigen::Vector3d& m,const Eigen::Matrix3d& rot, const Eigen::Vector3d& transl) const {
		Eigen::Vector3d Phi = reg_->phi(m);
		Eigen::Vector3d alpha = rot * Phi + transl;
		Eigen::Matrix<double, 3, 6> J;
		J.setZero(3, 6);

		// init d PhiInv / d Alpha
		Eigen::Matrix3d dPI_da = Eigen::Matrix3d::Zero();
		Eigen::Vector3d tmp = params_.K_ * alpha;
		double a2 = alpha(2) * alpha(2);
		dPI_da(0, 0) = params_.calibration_f_ / alpha(2);
		dPI_da(1, 1) = params_.calibration_f_ / alpha(2);
		tmp(0) /= -a2;
		tmp(1) /= -a2;
		tmp(2) /= -0.5 * alpha(2) * a2;
		tmp(0) += params_.calibration_c1_ / alpha(2);
		tmp(1) += params_.calibration_c2_ / alpha(2);
		tmp(2) += 1 / a2;
		dPI_da.block<3, 1>(0, 2) = tmp;

		Eigen::Matrix<double, 3, 3> da_dq;
		da_dq.block<3, 1>(0, 0) = dR_qx * rot * Phi;
		da_dq.block<3, 1>(0, 1) = dR_qy * rot * Phi;
		da_dq.block<3, 1>(0, 2) = dR_qz * rot * Phi;

		Eigen::Matrix<double, 3, 6> dh_dx;
		dh_dx.block<3, 3>(0, 0) = dPI_da;
		dh_dx.block<3, 3>(0, 3) = dPI_da * da_dq;

		return dh_dx;
	}

	// dh/dm
	inline Eigen::Matrix3d dh_dm(const Eigen::Vector3d& m, const Eigen::Matrix3d& rot, const Eigen::Vector3d& transl) const {

		const Eigen::Vector3d Phi = reg_->phi(m);
		const Eigen::Vector3d alpha = rot * Phi + transl;
//		Eigen::Matrix<double, 3, 6> J = Eigen::Matrix<double, 3, 6>::Zero();

		// init d PhiInv / d Alpha
		Eigen::Matrix3d dPI_da = Eigen::Matrix3d::Zero();
		Eigen::Vector3d tmp = params_.K_ * alpha;
		const double inv_a2 = 1.0 / alpha(2);
		const double inv_a22 = 1.0 / (alpha(2)*alpha(2));
//		double a2 = alpha(2) * alpha(2);
		dPI_da(0, 0) = params_.calibration_f_ * inv_a2;// / alpha(2);
		dPI_da(1, 1) = params_.calibration_f_ * inv_a2;// / alpha(2);
		tmp(0) *= -inv_a22;// /= -a2;
		tmp(1) *= -inv_a22;// /= -a2;
		tmp(2) *= -2.0 * inv_a2 * inv_a22;// /= -0.5 * alpha(2) * a2;
		tmp(0) += params_.calibration_c1_ * inv_a2;// / alpha(2);
		tmp(1) += params_.calibration_c2_ * inv_a2;// / alpha(2);
		tmp(2) += inv_a22;//1 / a2;
		dPI_da.block<3, 1>(0, 2) = tmp;

		Eigen::Matrix<double, 3, 3> da_dm = Eigen::Matrix<double, 3, 3>::Zero();
		const double inv_m22 = 1.0 / (m(2) * m(2));
		const double inv_m2 = 1.0 / m(2);
		const double inv_f = 1.0 / params_.calibration_f_;
		da_dm(0, 0) = inv_m2 * inv_f;// 1.0 / m(2) / params_.calibration_f_;
		da_dm(1, 1) = inv_m2 * inv_f;// 1.0 / m(2) / params_.calibration_f_;
		tmp(0) = -m(0) * inv_m22;
		tmp(1) = -m(1) * inv_m22;
		tmp(2) = -inv_m22;
		da_dm.block<3, 1>(0, 2) = params_.KInv_ * tmp;

		return dPI_da * da_dm;
	}

	GradientFunctorPointFeature(MultiResolutionColorSurfelMap* source,
			MultiResolutionColorSurfelMap* target,
			MultiResolutionColorSurfelRegistration::FeatureAssociationList* assocList,
			const MultiResolutionColorSurfelRegistration::Params& params,
			MultiResolutionColorSurfelRegistration* reg, double tx,
			double ty, double tz, double qx, double qy, double qz, double qw ) {

		source_ = source;
		target_ = target;
		assocList_ = assocList;
		params_ = params;
		reg_ = reg;

		const double inv_qw = 1.0 / qw;

		currentTransform.setIdentity();
		currentTransform.block<3, 3>(0, 0) = Eigen::Matrix3d(
				Eigen::Quaterniond(qw, qx, qy, qz));
		currentTransform(0, 3) = tx;
		currentTransform(1, 3) = ty;
		currentTransform(2, 3) = tz;


		dR_qx.setZero();
		dR_qx(1,2) = -2;
		dR_qx(2,1) = 2;

		dR_qy.setZero();
		dR_qy(0,2) = 2;
		dR_qy(2,0) = -2;

		dR_qz.setZero();
		dR_qz(0,1) = -2;
		dR_qz(1,0) = 2;

//		// matrix(
//		//  [ 0,
//		//    2*((qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qy),
//		//    2*(qz-(qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)) ],
//		//  [ 2*(qy-(qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)),
//		//    -4*qx,
//		//    2*(qx^2/sqrt(-qz^2-qy^2-qx^2+1)-sqrt(-qz^2-qy^2-qx^2+1)) ],
//		//  [ 2*((qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)+qz),
//		//    2*(sqrt(-qz^2-qy^2-qx^2+1)-qx^2/sqrt(-qz^2-qy^2-qx^2+1)),
//		//    -4*qx ]
//		// )
//		dR_qx(0, 0) = 0.0;
//		dR_qx(0, 1) = 2.0 * ((qx * qz) * inv_qw + qy);
//		dR_qx(0, 2) = 2.0 * (qz - (qx * qy) * inv_qw);
//		dR_qx(1, 0) = 2.0 * (qy - (qx * qz) * inv_qw);
//		dR_qx(1, 1) = -4.0 * qx;
//		dR_qx(1, 2) = 2.0 * (qx * qx * inv_qw - qw);
//		dR_qx(2, 0) = 2.0 * ((qx * qy) * inv_qw + qz);
//		dR_qx(2, 1) = 2.0 * (qw - qx * qx * inv_qw);
//		dR_qx(2, 2) = -4.0 * qx;
//
//		// matrix(
//		//  [ -4*qy,
//		//    2*((qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qx),
//		//    2*(sqrt(-qz^2-qy^2-qx^2+1)-qy^2/sqrt(-qz^2-qy^2-qx^2+1)) ],
//		//  [ 2*(qx-(qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)),
//		//    0,
//		//    2*((qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)+qz) ],
//		//  [ 2*(qy^2/sqrt(-qz^2-qy^2-qx^2+1)-sqrt(-qz^2-qy^2-qx^2+1)),
//		//    2*(qz-(qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)),
//		//    -4*qy ]
//		// )
//
//		dR_qy(0, 0) = -4.0 * qy;
//		dR_qy(0, 1) = 2.0 * ((qy * qz) * inv_qw + qx);
//		dR_qy(0, 2) = 2.0 * (qw - qy * qy * inv_qw);
//		dR_qy(1, 0) = 2.0 * (qx - (qy * qz) * inv_qw);
//		dR_qy(1, 1) = 0.0;
//		dR_qy(1, 2) = 2.0 * ((qx * qy) * inv_qw + qz);
//		dR_qy(2, 0) = 2.0 * (qy * qy * inv_qw - qw);
//		dR_qy(2, 1) = 2.0 * (qz - (qx * qy) * inv_qw);
//		dR_qy(2, 2) = -4.0 * qy;
//
//		// matrix(
//		//  [ -4*qz,
//		//    2*(qz^2/sqrt(-qz^2-qy^2-qx^2+1)-sqrt(-qz^2-qy^2-qx^2+1)),
//		//    2*(qx-(qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)) ],
//		//  [ 2*(sqrt(-qz^2-qy^2-qx^2+1)-qz^2/sqrt(-qz^2-qy^2-qx^2+1)),
//		//    -4*qz,
//		//    2*((qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qy) ],
//		//  [ 2*((qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qx),
//		//    2*(qy-(qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)),
//		//    0 ]
//		// )
//		dR_qz(0, 0) = -4.0 * qz;
//		dR_qz(0, 1) = 2.0 * (qz * qz * inv_qw - qw);
//		dR_qz(0, 2) = 2.0 * (qx - (qy * qz) * inv_qw);
//		dR_qz(1, 0) = 2.0 * (qw - qz * qz * inv_qw);
//		dR_qz(1, 1) = -4.0 * qz;
//		dR_qz(1, 2) = 2.0 * ((qx * qz) * inv_qw + qy);
//		dR_qz(2, 0) = 2.0 * ((qy * qz) * inv_qw + qx);
//		dR_qz(2, 1) = 2.0 * (qy - (qx * qz) * inv_qw);
//		dR_qz(2, 2) = 0.0;


	}

	~GradientFunctorPointFeature() {
	}


	double tx, ty, tz, qx, qy, qz, qw;
	Eigen::Matrix4d currentTransform;
	Eigen::Matrix3d dR_qx, dR_qy, dR_qz;

	MultiResolutionColorSurfelRegistration::FeatureAssociationList* assocList_;
	MultiResolutionColorSurfelRegistration::Params params_;

	MultiResolutionColorSurfelMap* source_;
	MultiResolutionColorSurfelMap* target_;

	MultiResolutionColorSurfelRegistration* reg_;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};


double MultiResolutionColorSurfelRegistration::preparePointFeatureDerivatives( const Eigen::Matrix<double, 6, 1>& x, double qw, double mahaldist ) {
	double error = 0;

	Eigen::Matrix3d id = Eigen::Matrix3d::Identity();
	Eigen::Vector3d null = Eigen::Vector3d::Zero();

	MRCSRFAL& fal = featureAssociations_;
	Eigen::Matrix3d rot = Eigen::Matrix3d(Eigen::Quaterniond(qw, x(3), x(4), x(5)));
	Eigen::Vector3d trnsl(x(0), x(1), x(2));
	GradientFunctorPointFeature gff(source_, target_, &fal, params_, this, x(0), x(1), x(2), x(3), x(4), x(5), qw);

	for (MRCSRFAL::iterator it = fal.begin(); it != fal.end(); ++it) {

		const PointFeature& f = source_->features_[it->src_idx_];
		const PointFeature& f2 = target_->features_[it->dst_idx_];

		it->match = 1;

		const Eigen::Vector3d lmPos = it->landmark_pos; // 3D pos in target frame
		const Eigen::Vector3d lmPosA = rot * lmPos + trnsl; // 3D pos in src frame
		const Eigen::Vector3d lm25D_A = phiInv(lmPosA);
		const Eigen::Vector3d lm25D_B = phiInv(lmPos);
		const Eigen::Matrix3d SigmaA = f.invzinvcov_;
		const Eigen::Matrix3d SigmaB = f2.invzinvcov_;
		const Eigen::Vector3d z25D_A = f.invzpos_;
		const Eigen::Vector3d z25D_B = f2.invzpos_;
		const Eigen::Vector3d diffA = -(z25D_A - lm25D_A);
		const Eigen::Vector3d diffB = -(z25D_B - lm25D_B);

		const Eigen::Vector3d SdiffA = SigmaA * diffA;
		const Eigen::Vector3d SdiffB = SigmaB * diffB;

		double src_error = diffA.transpose() * SdiffA;
		double dst_error = diffB.transpose() * SdiffB;

		if (src_error > mahaldist) {
			src_error = mahaldist;
			it->match = 0;
		}

		if (dst_error > mahaldist) {
			dst_error = mahaldist;
			it->match = 0;
		}

		error += (src_error + dst_error);

		if (!it->match)
			continue;

		const Eigen::Matrix<double, 3, 6> dhA_dx = gff.dh_dx(lm25D_B, rot, trnsl);
		const Eigen::Matrix3d dhA_dm = gff.dh_dm(lm25D_B, rot, trnsl);
		const Eigen::Matrix3d dhB_dm = gff.dh_dm(lm25D_B, id, null);

		const Eigen::Matrix3d dhA_dmS = dhA_dm.transpose() * SigmaA;
		const Eigen::Matrix3d dhB_dmS = dhB_dm.transpose() * SigmaB;
		const Eigen::Matrix<double, 6, 3> dhA_dx_S = dhA_dx.transpose() * SigmaA;

		it->Hpp = dhA_dx_S * dhA_dx;
		it->Hpl = dhB_dm.transpose() * SigmaA * dhA_dx;
		it->Hll = (dhA_dmS * dhA_dm + dhB_dmS * dhB_dm);
		it->bp = dhA_dx_S * diffA;
		it->bl = (dhA_dmS * diffA + dhB_dmS * diffB);

	}
	return error;
}


bool MultiResolutionColorSurfelRegistration::registrationErrorFunctionWithFirstDerivative( const Eigen::Matrix< double, 6, 1 >& x, double& f, Eigen::Matrix< double, 6, 1 >& df_dx, MultiResolutionColorSurfelRegistration::SurfelAssociationList& surfelAssociations ) {

	double sumError = 0.0;
	double sumWeight = 0.0;

	df_dx.setZero();

	const double tx = x( 0 );
	const double ty = x( 1 );
	const double tz = x( 2 );
	const double qx = x( 3 );
	const double qy = x( 4 );
	const double qz = x( 5 );
	if( qx*qx+qy*qy+qz*qz > 1.0 )
		std::cout << "quaternion not stable!!\n";
	const double qw = lastWSign_ * sqrtf(1.0-qx*qx-qy*qy-qz*qz); // retrieve sign from last qw

	GradientFunctor gf( &surfelAssociations, params_, tx, ty, tz, qx, qy, qz, qw, false, false, interpolate_neighbors_ );


	if( params_.parallel_ )
		tbb::parallel_for( tbb::blocked_range<size_t>(0,surfelAssociations.size()), gf );
	else
		std::for_each( surfelAssociations.begin(), surfelAssociations.end(), gf );


	double numMatches = 0;
	for( MultiResolutionColorSurfelRegistration::SurfelAssociationList::iterator it = surfelAssociations.begin(); it != surfelAssociations.end(); ++it ) {

		if( !it->match )
			continue;

		float nweight = it->n_src_->value_.assocWeight_ * it->n_dst_->value_.assocWeight_;
		float weight = nweight * it->weight;

		df_dx += weight * it->df_dx;
		sumError += weight * it->error;
		sumWeight += weight;
		numMatches += 1.0;//nweight;

	}

	if( sumWeight <= 1e-10 ) {
		sumError = std::numeric_limits<double>::max();
		return false;
	}
	else if( numMatches < params_.registration_min_num_surfels_ ) {
		sumError = std::numeric_limits<double>::max();
		return false;
	}
	else {
		sumError /= sumWeight;
		df_dx /= sumWeight;
	}

	if( params_.use_prior_pose_ ) {

		df_dx += 2.0 * params_.prior_pose_invcov_ * (x - params_.prior_pose_mean_);
	}

	f = sumError;
	return true;



}



bool MultiResolutionColorSurfelRegistration::registrationErrorFunctionWithFirstAndSecondDerivative( const Eigen::Matrix< double, 6, 1 >& x, bool relativeDerivative, double& f, Eigen::Matrix< double, 6, 1 >& df_dx, Eigen::Matrix< double, 6, 6 >& d2f_dx2, MultiResolutionColorSurfelRegistration::SurfelAssociationList& surfelAssociations ) {

	double sumError = 0.0;
	double sumWeight = 0.0;

	df_dx.setZero();
	d2f_dx2.setZero();

	const double tx = x( 0 );
	const double ty = x( 1 );
	const double tz = x( 2 );
	const double qx = x( 3 );
	const double qy = x( 4 );
	const double qz = x( 5 );
	if( qx*qx+qy*qy+qz*qz > 1.0 )
		std::cout << "quaternion not stable!!\n";
	const double qw = lastWSign_ * sqrtf(1.0-qx*qx-qy*qy-qz*qz); // retrieve sign from last qw

	GradientFunctor gf( &surfelAssociations, params_, tx, ty, tz, qx, qy, qz, qw, relativeDerivative, true, interpolate_neighbors_ );

	if( params_.parallel_ )
		tbb::parallel_for( tbb::blocked_range<size_t>(0,surfelAssociations.size()), gf );
	else
		std::for_each( surfelAssociations.begin(), surfelAssociations.end(), gf );

	int cidx = 0;
	if( correspondences_source_points_ ) {
		correspondences_source_points_->points.resize(surfelAssociations.size());
		correspondences_target_points_->points.resize(surfelAssociations.size());
	}


	double numMatches = 0;
	for( MultiResolutionColorSurfelRegistration::SurfelAssociationList::iterator it = surfelAssociations.begin(); it != surfelAssociations.end(); ++it ) {

		if( !it->match )
			continue;


		float nweight = it->n_src_->value_.assocWeight_ * it->n_dst_->value_.assocWeight_;
		float weight = nweight * it->weight;

		df_dx += weight * it->df_dx;
		d2f_dx2 += weight * it->d2f;
		sumError += weight * it->error;
		sumWeight += weight;
		numMatches += 1.0;//nweight;



		if( correspondences_source_points_ ) {

			pcl::PointXYZRGB& p1 = correspondences_source_points_->points[cidx];
			pcl::PointXYZRGB& p2 = correspondences_target_points_->points[cidx];

			Eigen::Vector4f pos1 = it->n_dst_->getCenterPosition();
			Eigen::Vector4f pos2 = it->n_src_->getCenterPosition();

			p1.x = pos1(0);
			p1.y = pos1(1);
			p1.z = pos1(2);


//			p1.x = it->dst_->mean_[0];
//			p1.y = it->dst_->mean_[1];
//			p1.z = it->dst_->mean_[2];

			p1.r = nweight * 255.f;
			p1.g = 0;
			p1.b = (1.f-nweight) * 255.f;

			Eigen::Vector4d pos;
			pos.block<3,1>(0,0) = pos2.block<3,1>(0,0).cast<double>();
//			pos.block<3,1>(0,0) = it->src_->mean_.block<3,1>(0,0);
			pos(3,0) = 1.f;

			const Eigen::Vector4d pos_src = gf.currentTransform * pos;

			p2.x = pos_src[0];
			p2.y = pos_src[1];
			p2.z = pos_src[2];

			p2.r = nweight * 255.f;
			p2.g = 0;
			p2.b = (1.f-nweight) * 255.f;

			cidx++;
		}

	}


	if( correspondences_source_points_ ) {
		correspondences_source_points_->points.resize(cidx);
		correspondences_target_points_->points.resize(cidx);
	}

	if( sumWeight <= 1e-10 ) {
		sumError = std::numeric_limits<double>::max();
//			ROS_INFO("no surfel match!");
		return false;
	}
	else if( numMatches < params_.registration_min_num_surfels_ ) {
		sumError = std::numeric_limits<double>::max();
		std::cout << "not enough surfels for robust matching " << numMatches << "\n";
		return false;
	}
	else {
		sumError /= sumWeight;
		df_dx /= sumWeight;
		d2f_dx2 /= sumWeight;
	}


	if( params_.use_prior_pose_ ) {

		df_dx += 2.0 * params_.prior_pose_invcov_ * (x - params_.prior_pose_mean_);
		d2f_dx2 += 2.0 * params_.prior_pose_invcov_;
	}



	f = sumError;
	return true;

}



bool MultiResolutionColorSurfelRegistration::registrationErrorFunctionLM( const Eigen::Matrix< double, 6, 1 >& x, double& f, MultiResolutionColorSurfelRegistration::SurfelAssociationList& surfelAssociations, MultiResolutionColorSurfelRegistration::FeatureAssociationList& featureAssociations, double mahaldist ) {

	double sumFeatureError	= 0.0;
	double sumFeatureWeight = 0.0;
	double sumSurfelError	= 0.0;
	double sumSurfelWeight	= 0.0;

	const double tx = x( 0 );
	const double ty = x( 1 );
	const double tz = x( 2 );
	const double qx = x( 3 );
	const double qy = x( 4 );
	const double qz = x( 5 );
	if( qx*qx+qy*qy+qz*qz > 1.0 )
		std::cout << "quaternion not stable!!\n";
	const double qw = lastWSign_ * sqrtf(1.0-qx*qx-qy*qy-qz*qz); // retrieve sign from last qw

	if (params_.registerSurfels_) {

		GradientFunctorLM gf( &surfelAssociations, params_, tx, ty, tz, qx, qy, qz, qw, false );

		if( params_.parallel_ )
			tbb::parallel_for( tbb::blocked_range<size_t>(0,surfelAssociations.size()), gf );
		else
			std::for_each( surfelAssociations.begin(), surfelAssociations.end(), gf );

		int cidx = 0;
		if( correspondences_source_points_ ) {
			correspondences_source_points_->points.resize(surfelAssociations.size());
			correspondences_target_points_->points.resize(surfelAssociations.size());
		}


		double numMatches = 0;
		for( MultiResolutionColorSurfelRegistration::SurfelAssociationList::iterator it = surfelAssociations.begin(); it != surfelAssociations.end(); ++it ) {

			if( !it->match )
				continue;


			float nweight = it->n_src_->value_.assocWeight_ * it->n_dst_->value_.assocWeight_;
			float weight = nweight * it->weight;

			sumSurfelError += weight * it->error;
			sumSurfelWeight += weight;
			numMatches += 1.0;//nweight;



			if( correspondences_source_points_ ) {

				pcl::PointXYZRGB& p1 = correspondences_source_points_->points[cidx];
				pcl::PointXYZRGB& p2 = correspondences_target_points_->points[cidx];

				Eigen::Vector4f pos1 = it->n_dst_->getCenterPosition();
				Eigen::Vector4f pos2 = it->n_src_->getCenterPosition();

				p1.x = pos1(0);
				p1.y = pos1(1);
				p1.z = pos1(2);


	//			p1.x = it->dst_->mean_[0];
	//			p1.y = it->dst_->mean_[1];
	//			p1.z = it->dst_->mean_[2];

				p1.r = nweight * 255.f;
				p1.g = 0;
				p1.b = (1.f-nweight) * 255.f;

				Eigen::Vector4d pos;
				pos.block<3,1>(0,0) = pos2.block<3,1>(0,0).cast<double>();
	//			pos.block<3,1>(0,0) = it->src_->mean_.block<3,1>(0,0);
				pos(3,0) = 1.f;

				const Eigen::Vector4d pos_src = gf.currentTransform * pos;

				p2.x = pos_src[0];
				p2.y = pos_src[1];
				p2.z = pos_src[2];

				p2.r = nweight * 255.f;
				p2.g = 0;
				p2.b = (1.f-nweight) * 255.f;

				cidx++;
			}

		}


		if( correspondences_source_points_ ) {
			correspondences_source_points_->points.resize(cidx);
			correspondences_target_points_->points.resize(cidx);
		}

		if( sumSurfelWeight <= 1e-10 ) {
			sumSurfelError = std::numeric_limits<double>::max();
	//			ROS_INFO("no surfel match!");
			return false;
		}
		else if( numMatches < params_.registration_min_num_surfels_ ) {
			sumSurfelError = std::numeric_limits<double>::max();
			std::cout << "not enough surfels for robust matching " << numMatches << "\n";
			return false;
		}


		f = sumSurfelError / sumSurfelWeight * numMatches;

	}

	if (params_.registerFeatures_) {

		Eigen::Matrix3d rot = Eigen::Matrix3d(Eigen::Quaterniond(qw, qx, qy, qz));
		Eigen::Vector3d trnsl(tx, ty, tz);
		Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
		transform.block<3, 3>(0, 0) = rot;
		transform.block<3, 1>(0, 3) = trnsl;

		for (MRCSReg::FeatureAssociationList::iterator it = featureAssociations.begin(); it != featureAssociations.end(); ++it) {

//			if( !it->match )
//				continue;
			it->match = 1;

			Eigen::Vector3d src = source_->features_[it->src_idx_].invzpos_.block<3, 1>(0,0);
			Eigen::Vector3d dst = target_->features_[it->dst_idx_].invzpos_.block<3, 1>(0,0);

			Eigen::Vector3d lm = it->landmark_pos;

			double src_error = (src - phiInv(rot * lm + trnsl)).transpose() * source_->features_[it->src_idx_].invzcov_.block<3, 3>(0,0).inverse() * (src - phiInv(rot * lm + trnsl));
			double dst_error = (dst - phiInv(lm)).transpose() * target_->features_[it->dst_idx_].invzcov_.block<3, 3>(0,0).inverse() * (dst - phiInv(lm));

			if (src_error > mahaldist) {
				it->match = 0;
				src_error = mahaldist;
			}

			if (dst_error > mahaldist) {
				it->match = 0;
				dst_error = mahaldist;
			}

			it->error = src_error + dst_error;
			sumFeatureError += it->error;
			sumFeatureWeight += 1.0;
		}
		f += params_.pointFeatureWeight_ * sumFeatureError;// / (double) featureAssociations.size();
	}

	if( params_.use_prior_pose_ ) {
		f += (params_.prior_pose_mean_ - x).transpose() * params_.prior_pose_invcov_ * (params_.prior_pose_mean_ - x);
	}


	return true;

}



bool MultiResolutionColorSurfelRegistration::registrationSurfelErrorFunctionWithFirstAndSecondDerivativeLM( const Eigen::Matrix< double, 6, 1 >& x, double& f, Eigen::Matrix< double, 6, 1 >& df, Eigen::Matrix< double, 6, 6 >& d2f, MultiResolutionColorSurfelRegistration::SurfelAssociationList& surfelAssociations ) {

	double sumError = 0.0;
	double sumWeight = 0.0;

	df.setZero();
	d2f.setZero();

	const double tx = x( 0 );
	const double ty = x( 1 );
	const double tz = x( 2 );
	const double qx = x( 3 );
	const double qy = x( 4 );
	const double qz = x( 5 );
	if( qx*qx+qy*qy+qz*qz > 1.0 )
		std::cout << "quaternion not stable!!\n";
	const double qw = lastWSign_ * sqrtf(1.0-qx*qx-qy*qy-qz*qz); // retrieve sign from last qw

	GradientFunctorLM gf( &surfelAssociations, params_, tx, ty, tz, qx, qy, qz, qw, true, false, interpolate_neighbors_ );

	if( params_.parallel_ )
		tbb::parallel_for( tbb::blocked_range<size_t>(0,surfelAssociations.size()), gf );
	else
		std::for_each( surfelAssociations.begin(), surfelAssociations.end(), gf );

	int cidx = 0;
	if( correspondences_source_points_ ) {
		correspondences_source_points_->points.resize(surfelAssociations.size());
		correspondences_target_points_->points.resize(surfelAssociations.size());
	}


	double numMatches = 0;
	for( MultiResolutionColorSurfelRegistration::SurfelAssociationList::iterator it = surfelAssociations.begin(); it != surfelAssociations.end(); ++it ) {

		if( !it->match )
			continue;


		float nweight = it->n_src_->value_.assocWeight_ * it->n_dst_->value_.assocWeight_;
		float weight = nweight * it->weight;

		const Eigen::Matrix< double, 6, 3 > JtW = weight * it->dh_dx.transpose() * it->W;

		df += JtW * (it->z - it->h);
		d2f += JtW * it->dh_dx;

		sumError += weight * it->error;
		sumWeight += weight;
		numMatches += 1.0;//nweight;



		if( correspondences_source_points_ ) {

			pcl::PointXYZRGB& p1 = correspondences_source_points_->points[cidx];
			pcl::PointXYZRGB& p2 = correspondences_target_points_->points[cidx];

			Eigen::Vector4f pos1 = it->n_dst_->getCenterPosition();
			Eigen::Vector4f pos2 = it->n_src_->getCenterPosition();

			p1.x = pos1(0);
			p1.y = pos1(1);
			p1.z = pos1(2);


//			p1.x = it->dst_->mean_[0];
//			p1.y = it->dst_->mean_[1];
//			p1.z = it->dst_->mean_[2];

			p1.r = nweight * 255.f;
			p1.g = 0;
			p1.b = (1.f-nweight) * 255.f;

			Eigen::Vector4d pos;
			pos.block<3,1>(0,0) = pos2.block<3,1>(0,0).cast<double>();
//			pos.block<3,1>(0,0) = it->src_->mean_.block<3,1>(0,0);
			pos(3,0) = 1.f;

			const Eigen::Vector4d pos_src = gf.currentTransform * pos;

			p2.x = pos_src[0];
			p2.y = pos_src[1];
			p2.z = pos_src[2];

			p2.r = nweight * 255.f;
			p2.g = 0;
			p2.b = (1.f-nweight) * 255.f;

			cidx++;
		}

	}


	if( correspondences_source_points_ ) {
		correspondences_source_points_->points.resize(cidx);
		correspondences_target_points_->points.resize(cidx);
	}

	if( sumWeight <= 1e-10 ) {
		sumError = std::numeric_limits<double>::max();
//			ROS_INFO("no surfel match!");
		return false;
	}
	else if( numMatches < params_.registration_min_num_surfels_ ) {
		sumError = std::numeric_limits<double>::max();
		std::cout << "not enough surfels for robust matching " << numMatches << "\n";
		return false;
	}

	f = sumError / sumWeight * numMatches;
	df = df / sumWeight * numMatches;
	d2f = d2f / sumWeight * numMatches;


	if( params_.use_prior_pose_ ) {

		f += (x - params_.prior_pose_mean_).transpose() * params_.prior_pose_invcov_ * (x - params_.prior_pose_mean_);
		df += params_.prior_pose_invcov_ * (params_.prior_pose_mean_ - x);
		d2f += params_.prior_pose_invcov_;

	}




	return true;

}



bool MultiResolutionColorSurfelRegistration::estimateTransformationNewton( Eigen::Matrix4d& transform, int coarseToFineIterations, int fineIterations ) {

	Eigen::Matrix4d initialTransform = transform;

	// coarse alignment with features
	// fine alignment without features

	float minResolution = std::min( params_.startResolution_, params_.stopResolution_ );
	float maxResolution = std::max( params_.startResolution_, params_.stopResolution_ );

	const double step_max = 0.1;
	const double step_size_coarse = 1.0;
	const double step_size_fine = 1.0;

	Eigen::Matrix4d currentTransform = transform;

	const int maxIterations = coarseToFineIterations + fineIterations;


	Eigen::Matrix< double, 6, 1 > x, last_x, df, best_x, best_g;
	Eigen::Matrix< double, 6, 6 > d2f;

	// initialize with current transform
	Eigen::Quaterniond q( currentTransform.block<3,3>(0,0) );

	x(0) = currentTransform( 0, 3 );
	x(1) = currentTransform( 1, 3 );
	x(2) = currentTransform( 2, 3 );
	x(3) = q.x();
	x(4) = q.y();
	x(5) = q.z();
	lastWSign_ = q.w() / fabsf(q.w());


	last_x = x;


	target_->clearAssociations();

	double best_f = std::numeric_limits<double>::max();
	Eigen::Matrix4d bestTransform;
	bestTransform.setIdentity();
	best_x = x;
	best_g.setZero();


	pcl::StopWatch stopwatch;

	transform.setIdentity();
	const double qx = x( 3 );
	const double qy = x( 4 );
	const double qz = x( 5 );
	transform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( lastWSign_*sqrt(1.0-qx*qx-qy*qy-qz*qz), qx, qy, qz ) );
	transform(0,3) = x( 0 );
	transform(1,3) = x( 1 );
	transform(2,3) = x( 2 );

	double associateTime = 0;
	double gradientTime = 0;

	MultiResolutionColorSurfelRegistration::SurfelAssociationList surfelAssociations;

	bool retVal = true;


	int iter = 0;
	while( iter < maxIterations ) {


		// stays at minresolution after coarseToFineIterations
		float searchDistFactor = 2.f;//std::max( 1.f, 1.f + 1.f * (((float)(fineIterations / 2 - iter)) / (float)(fineIterations / 2)) );
		float maxSearchDist = 2.f*maxResolution;//(minResolution + (maxResolution-minResolution) * ((float)(maxIterations - iter)) / (float)maxIterations);

		MultiResolutionColorSurfelRegistration::SurfelAssociationList surfelAssociations;
		if( iter < coarseToFineIterations ) {
			stopwatch.reset();
			associateMapsBreadthFirstParallel( surfelAssociations, *source_, *target_, targetSamplingMap_, transform, 0.99f*minResolution, 1.01f*maxResolution, searchDistFactor, maxSearchDist, params_.use_features_ );
			double deltat = stopwatch.getTimeSeconds() * 1000.0;
			associateTime += deltat;
			interpolate_neighbors_ = false;

		}
		else {
			if( iter == coarseToFineIterations ) {
				target_->clearAssociations();
			}

			stopwatch.reset();
			associateMapsBreadthFirstParallel( surfelAssociations, *source_, *target_, targetSamplingMap_, transform, 0.99f*minResolution, 1.01f*maxResolution, searchDistFactor, maxSearchDist, false );
			double deltat = stopwatch.getTimeSeconds() * 1000.0;
			associateTime += deltat;
			interpolate_neighbors_ = true;
		}


		// evaluate function and derivative
		double f = 0.0;
		stopwatch.reset();
		retVal = registrationErrorFunctionWithFirstAndSecondDerivative( x, true, f, df, d2f, surfelAssociations );

		if( !retVal ) {
			df.setZero();
			d2f.setIdentity();
		}

		double deltat2 = stopwatch.getTimeSeconds() * 1000.0;
		gradientTime += deltat2;

		if( f < best_f ) {
			best_f = f;
			bestTransform = transform;
		}



		Eigen::Matrix< double, 6, 1 > lastX = x;
		Eigen::Matrix< double, 6, 6 > d2f_inv;
		d2f_inv.setZero();
		if( fabsf( d2f.determinant() ) > std::numeric_limits<double>::epsilon() ) {

			double step_size_i = step_size_fine;

			d2f_inv = d2f.inverse();
			Eigen::Matrix< double, 6, 1 > deltaX = -step_size_i * d2f_inv * df;

			last_x = x;


			double qx = x( 3 );
			double qy = x( 4 );
			double qz = x( 5 );
			double qw = lastWSign_*sqrt(1.0-qx*qx-qy*qy-qz*qz);

			currentTransform.setIdentity();
			currentTransform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
			currentTransform(0,3) = x( 0 );
			currentTransform(1,3) = x( 1 );
			currentTransform(2,3) = x( 2 );


			qx = deltaX( 3 );
			qy = deltaX( 4 );
			qz = deltaX( 5 );
			qw = sqrt(1.0-qx*qx-qy*qy-qz*qz);

			Eigen::Matrix4d deltaTransform = Eigen::Matrix4d::Identity();
			deltaTransform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
			deltaTransform(0,3) = deltaX( 0 );
			deltaTransform(1,3) = deltaX( 1 );
			deltaTransform(2,3) = deltaX( 2 );

			Eigen::Matrix4d newTransform = deltaTransform * currentTransform;

			x( 0 ) = newTransform(0,3);
			x( 1 ) = newTransform(1,3);
			x( 2 ) = newTransform(2,3);

			Eigen::Quaterniond q_new( newTransform.block<3,3>(0,0) );
			x( 3 ) = q_new.x();
			x( 4 ) = q_new.y();
			x( 5 ) = q_new.z();

		}


		double qx = x( 3 );
		double qy = x( 4 );
		double qz = x( 5 );
		double qw = lastWSign_*sqrt(1.0-qx*qx-qy*qy-qz*qz);



		if( isnan(qw) || fabsf(qx) > 1.f || fabsf(qy) > 1.f || fabsf(qz) > 1.f ) {
			x = last_x;
			return false;
		}


		transform.setIdentity();
		transform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
		transform(0,3) = x( 0 );
		transform(1,3) = x( 1 );
		transform(2,3) = x( 2 );


		iter++;

	}


	return retVal;


}





bool MultiResolutionColorSurfelRegistration::estimateTransformationLevenbergMarquardt( Eigen::Matrix4d& transform, int maxIterations ) {

	const bool useFeatures = params_.use_features_;

	const double tau = 10e-5;
//	const double min_gradient_size = 1e-4;
	const double min_delta = 1e-3; // was 1e-3
	const double min_error = 1e-6;

	Eigen::Matrix4d initialTransform = transform;

	float minResolution = std::min( params_.startResolution_, params_.stopResolution_ );
	float maxResolution = std::max( params_.startResolution_, params_.stopResolution_ );

	Eigen::Matrix4d currentTransform = transform;



	// initialize with current transform
	Eigen::Matrix< double, 6, 1 > x;
	Eigen::Quaterniond q( currentTransform.block<3,3>(0,0) );

	x(0) = currentTransform( 0, 3 );
	x(1) = currentTransform( 1, 3 );
	x(2) = currentTransform( 2, 3 );
	x(3) = q.x();
	x(4) = q.y();
	x(5) = q.z();
	lastWSign_ = q.w() / fabsf(q.w());


	pcl::StopWatch stopwatch;

	Eigen::Matrix< double, 6, 1 > df;
	Eigen::Matrix< double, 6, 6 > d2f;

	const Eigen::Matrix< double, 6, 6 > id6 = Eigen::Matrix< double, 6, 6 >::Identity();
	double mu = -1.0;
	double nu = 2;

	double last_error = std::numeric_limits<double>::max();

	MultiResolutionColorSurfelRegistration::SurfelAssociationList surfelAssociations;

	bool reassociate = true;

	bool reevaluateGradient = true;

	bool retVal = true;

	int iter = 0;
	while( iter < maxIterations ) {

		if( reevaluateGradient ) {

			if( reassociate ) {
				target_->clearAssociations();
			}

			float searchDistFactor = 2.f;
			float maxSearchDist = 2.f*maxResolution;

			stopwatch.reset();
			surfelAssociations.clear();
			associateMapsBreadthFirstParallel( surfelAssociations, *source_, *target_, targetSamplingMap_, transform, 0.99f*minResolution, 1.01f*maxResolution, searchDistFactor, maxSearchDist, useFeatures );
			double deltat = stopwatch.getTime();
//			std::cout << "assoc took: " << deltat << "\n";

			interpolate_neighbors_ = false;

			stopwatch.reset();
			retVal = registrationSurfelErrorFunctionWithFirstAndSecondDerivativeLM( x, last_error, df, d2f, surfelAssociations );
			double deltat2 = stopwatch.getTime();
//			std::cout << "reg deriv took: " << deltat2 << "\n";


		}

		reevaluateGradient = false;

		if( !retVal ) {
			std::cout << "registration failed\n";
			return false;
		}

//		double gradient_size = std::max( df.maxCoeff(), -df.minCoeff() );
//		if( last_error < min_error ) {
////			std::cout << "converged\n";
//			break;
//		}


		if( mu < 0 ) {
			mu = tau * std::max( d2f.maxCoeff(), -d2f.minCoeff() );
		}

//		std::cout << "mu: " << mu << "\n";


		Eigen::Matrix< double, 6, 1 > delta_x = Eigen::Matrix< double, 6, 1 >::Zero();
		Eigen::Matrix< double, 6, 6 > d2f_inv = Eigen::Matrix< double, 6, 6 >::Zero();
		if( fabsf( d2f.determinant() ) > std::numeric_limits<double>::epsilon() ) {

			d2f_inv = (d2f + mu * id6).inverse();

			delta_x = d2f_inv * df;

		}

		if( delta_x.norm() < min_delta ) {

			if( reassociate )
				break;

			reassociate = true;
			reevaluateGradient = true;
//			std::cout << "reassociating!\n";
		}
		else
			reassociate = false;


		double qx = x( 3 );
		double qy = x( 4 );
		double qz = x( 5 );
		double qw = lastWSign_*sqrt(1.0-qx*qx-qy*qy-qz*qz);


		currentTransform.setIdentity();
		currentTransform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
		currentTransform(0,3) = x( 0 );
		currentTransform(1,3) = x( 1 );
		currentTransform(2,3) = x( 2 );


		qx = delta_x( 3 );
		qy = delta_x( 4 );
		qz = delta_x( 5 );
		qw = sqrt(1.0-qx*qx-qy*qy-qz*qz);

		Eigen::Matrix4d deltaTransform = Eigen::Matrix4d::Identity();
		deltaTransform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
		deltaTransform(0,3) = delta_x( 0 );
		deltaTransform(1,3) = delta_x( 1 );
		deltaTransform(2,3) = delta_x( 2 );

		Eigen::Matrix4d newTransform = deltaTransform * currentTransform;


//		Eigen::Matrix< double, 6, 1 > x_new = x + delta_x;
		Eigen::Matrix< double, 6, 1 > x_new;
		x_new( 0 ) = newTransform(0,3);
		x_new( 1 ) = newTransform(1,3);
		x_new( 2 ) = newTransform(2,3);

		Eigen::Quaterniond q_new( newTransform.block<3,3>(0,0) );
		x_new( 3 ) = q_new.x();
		x_new( 4 ) = q_new.y();
		x_new( 5 ) = q_new.z();


//		std::cout << "iter: " << iter << ": " << delta_x.norm() << "\n";

//		std::cout << x_new.transpose() << "\n";
//
		double new_error = 0.0;
		featureAssociations_.clear();
		bool retVal2 = registrationErrorFunctionLM( x_new, new_error, surfelAssociations, featureAssociations_, 0 );

		if( !retVal2 )
			return false;

		double rho = (last_error - new_error) / (delta_x.transpose() * (mu * delta_x + df));


		if( rho > 0 ) {

			x = x_new;

			mu *= std::max( 0.333, 1.0 - pow( 2.0*rho-1.0, 3.0 ) );
			nu = 2;

			reevaluateGradient = true;

		}
		else {

			mu *= nu; nu *= 2.0;

		}



		qx = x( 3 );
		qy = x( 4 );
		qz = x( 5 );
		qw = lastWSign_*sqrt(1.0-qx*qx-qy*qy-qz*qz);



		if( isnan(qw) || fabsf(qx) > 1.f || fabsf(qy) > 1.f || fabsf(qz) > 1.f ) {
			return false;
		}


		transform.setIdentity();
		transform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
		transform(0,3) = x( 0 );
		transform(1,3) = x( 1 );
		transform(2,3) = x( 2 );


//		last_error = new_error;

		iter++;

	}


	return retVal;

}


bool MultiResolutionColorSurfelRegistration::estimateTransformationLevenbergMarquardtPF( Eigen::Matrix4d& transform, int maxIterations, double featureAssocMahalDist, double minDelta, double& mu, double& nu ) {

	const double tau = 1e-4;
	const double min_delta = minDelta;//1e-4;
	const double min_error = 1e-6;


	for (MRCSRFAL::iterator it = featureAssociations_.begin(); it != featureAssociations_.end(); ++it) {
		it->landmark_pos = target_->features_[it->dst_idx_].pos_.block<3, 1>(0,0);
	}

	float minResolution = std::min( params_.startResolution_, params_.stopResolution_ );
	float maxResolution = std::max( params_.startResolution_, params_.stopResolution_ );

	MultiResolutionColorSurfelRegistration::SurfelAssociationList surfelAssociations;


	Eigen::Matrix4d initialTransform = transform;
	Eigen::Matrix4d currentTransform = transform;

	// initialize with current transform
	Eigen::Matrix<double, 6, 1> x;
	Eigen::Quaterniond q(currentTransform.block<3, 3>(0, 0));

	x(0) = currentTransform(0, 3);
	x(1) = currentTransform(1, 3);
	x(2) = currentTransform(2, 3);
	x(3) = q.x();
	x(4) = q.y();
	x(5) = q.z();
	lastWSign_ = q.w() / fabsf(q.w());

	pcl::StopWatch stopwatch;

	Eigen::Matrix<double, 6, 6> compactH	= Eigen::Matrix<double, 6, 6>::Zero();
	Eigen::Matrix<double, 6, 1> rightSide	= Eigen::Matrix<double, 6, 1>::Zero();
	Eigen::Matrix<double, 6, 1> poseOFdf	= Eigen::Matrix<double, 6, 1>::Zero(); // df.block<6,1>(0,0), df = J^T * Sigma * diff

	const Eigen::Matrix<double, 6, 6> id6	= Eigen::Matrix<double, 6, 6>::Identity();
//	double mu = -1.0;
//	double nu = 2;

	double last_error = std::numeric_limits<double>::max();
	double new_error = 0.0;

	bool reassociate = true;

	bool reevaluateGradient = true;

	bool retVal = true;

	int iter = 0;
	while (iter < maxIterations) {

		if( params_.debugFeatures_ ) {
			// AreNo - test des BundleAdjustment
			cv::Mat sourceFrame = source_->img_rgb_.clone();
			cv::Mat targetFrame = target_->img_rgb_.clone();
			cv::Scalar color = 0;
			color.val[0] = 255;		// Blau
			cv::Scalar color2 = 0;
			color2.val[1] = 255;	// Grün
			cv::Scalar colorErr = 0;// schwarz
			cv::Scalar rot( 0, 0, 255, 0);
			Eigen::Matrix4d backTrnsf = transform.inverse();

			for( MRCSRFAL::iterator it = featureAssociations_.begin(); it != featureAssociations_.end(); ++it ) {

				if( !it->match )
					continue;

				Eigen::Vector3d dst = target_->features_[ it->dst_idx_ ].invzpos_.block<3,1>(0,0);
				Eigen::Vector3d src = source_->features_[ it->src_idx_ ].invzpos_.block<3,1>(0,0);
		//			Eigen::Vector3d src = h( dst, backTrnsf.block<3,3>(0,0), backTrnsf.block<3,1>(0,3));
				cv::Point srcPoint( src(0) , src(1) );
				cv::Point dstPoint( dst(0) , dst(1) );

				// LM-Messungen
				cv::circle( sourceFrame, srcPoint, 2, color, 2);
				cv::circle( targetFrame, dstPoint, 2, color, 2);
	//			cv::line( frameA, srcPoint, dstPoint, rot, 1, 0, 0 );


				// LM-Schätzungen
				Eigen::Vector3d pixA = phiInv( transform.block<3,3>(0,0) * it->landmark_pos + transform.block<3,1>(0,3) );
				Eigen::Vector3d pixB = phiInv( it->landmark_pos );
				cv::Point srcLMPoint( pixA(0) , pixA(1) );
				cv::Point dstLMPoint( pixB(0) , pixB(1) );

				if ( (pixA - src).block<2,1>(0,0).norm() > 10 )
				{
					cv::circle( sourceFrame, srcLMPoint, 4, colorErr, 2);
					cv::line( sourceFrame, srcLMPoint, srcPoint, colorErr, 1, 0, 0 );
				}
				else
					cv::circle( sourceFrame, srcLMPoint, 4, color2, 2);

				if ( (pixB - dst).block<2,1>(0,0).norm() > 10 )
				{
					cv::circle( targetFrame, dstLMPoint, 4, colorErr, 2);
					cv::line( targetFrame, dstLMPoint, dstPoint, colorErr, 1, 0, 0 );
				}
				else
					cv::circle( targetFrame, dstLMPoint, 4, color2, 2);
			}
			cv::imshow( "TargetFrame", targetFrame);
			cv::imshow( "SourceFrame", sourceFrame);
			cv::waitKey(10);
//			while( 	cv::waitKey(10) == -1 );
		}


		Eigen::Matrix<double, 6, 1> deltaS		= Eigen::Matrix<double, 6, 1>::Zero();
		Eigen::Matrix<double, 6, 6> surfeld2f	= Eigen::Matrix<double, 6, 6>::Zero();
		Eigen::Matrix<double, 6, 1> surfeldf	= Eigen::Matrix<double, 6, 1>::Zero();

//		double coarsefactor = 1.0 - (double)iter / (double)maxIterations;
//		double featureAssocMahalDist = params_.pointFeatureMatchingFineImagePosMahalDist_ + coarsefactor * (params_.pointFeatureMatchingCoarseImagePosMahalDist_ - params_.pointFeatureMatchingFineImagePosMahalDist_);

//		int numMatches = 0;
//		for (MRCSRFAL::iterator it = featureAssociations_.begin(); it != featureAssociations_.end(); ++it)
//		{
//			if (it->match == 0)
//				continue;
//
//			numMatches++;
//		}
//
//		double unmatchedFraction = 1.0 - (double)numMatches / (double)featureAssociations_.size();
//		double featureAssocMahalDist = params_.pointFeatureMatchingFineImagePosMahalDist_ + unmatchedFraction * (params_.pointFeatureMatchingCoarseImagePosMahalDist_ - params_.pointFeatureMatchingFineImagePosMahalDist_);
//
//		std::cout << iter << " " << featureAssocMahalDist << "\n";

		if (reevaluateGradient) {

			if (reassociate) {
				target_->clearAssociations();
			}

			stopwatch.reset();

			last_error = 0.0;
			compactH.setZero();
			rightSide.setZero();
			poseOFdf.setZero();
			if (params_.registerFeatures_)
			{
				stopwatch.reset();
				last_error = preparePointFeatureDerivatives(x, q.w(), featureAssocMahalDist);

				int numMatches = 0;
				for (MRCSRFAL::iterator it = featureAssociations_.begin(); it != featureAssociations_.end(); ++it)
				{
					if (it->match == 0)
						continue;

					numMatches++;

					Eigen::Matrix<double, 6, 3> tmp = it->Hpl.transpose() * it->Hll.inverse();
					compactH += it->Hpp - tmp * it->Hpl;
					rightSide += tmp * it->bl - it->bp;
					poseOFdf += -it->bp;

				}

				last_error *= params_.pointFeatureWeight_;
				compactH *= params_.pointFeatureWeight_;
				rightSide *= params_.pointFeatureWeight_;
				poseOFdf *= params_.pointFeatureWeight_;

				if( params_.debugFeatures_ )
					std::cout << "matched " << numMatches << "/" << featureAssociations_.size() << " associated pointfeatures\n";

				if( params_.debugFeatures_ )
					std::cout << "feature preparation took: " << stopwatch.getTime() << "\n";
			}

			if (params_.registerSurfels_)
			{
				float searchDistFactor = 2.f;
				float maxSearchDist = 2.f * maxResolution;

				stopwatch.reset();

				surfelAssociations.clear();
				associateMapsBreadthFirstParallel( surfelAssociations, *source_, *target_, targetSamplingMap_, transform, minResolution, maxResolution, searchDistFactor, maxSearchDist, false);

				double surfelError = 0;
				double deltat = stopwatch.getTime();
				// std::cout << "assoc took: " << deltat << "\n";

				interpolate_neighbors_ = true;

				if (!registrationSurfelErrorFunctionWithFirstAndSecondDerivativeLM( x, surfelError, surfeldf, surfeld2f, surfelAssociations))
				{
					std::cout << "Surfelregistration failed ------\n";
				}
				else
				{
					compactH += surfeld2f;
					rightSide += surfeldf;
					last_error += surfelError;
					poseOFdf += surfeldf;
				}
			}

		}

		reevaluateGradient = false;

		if (!retVal) {
			std::cout << "registration failed\n";
			return false;
		}

		if (mu < 0) {
			mu = tau * std::max(compactH.maxCoeff(), -compactH.minCoeff());
		}

		Eigen::Matrix<double, 6, 1> delta_x	= Eigen::Matrix<double, 6, 1>::Zero();
		Eigen::Matrix<double, 6, 6> d2f		= compactH + mu * Eigen::Matrix<double, 6, 6>::Identity();

		// delta_x für feature
		if (fabsf(d2f.determinant()) > std::numeric_limits<double>::epsilon())
		{
			delta_x = d2f.inverse() * rightSide;
		}
		else {
			std::cout << "Det(d2f) =\t" << d2f.determinant() << "\n";
		}


		if (delta_x.norm() < min_delta) {

			if (reassociate) {
				break;
			}

			reassociate = true;
			reevaluateGradient = true;
		} else
			reassociate = false;


		double qx = x( 3 );
		double qy = x( 4 );
		double qz = x( 5 );
		double qw = lastWSign_*sqrt(1.0-qx*qx-qy*qy-qz*qz);

		Eigen::Matrix4d currentTransform;
		currentTransform.setIdentity();
		currentTransform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
		currentTransform(0,3) = x( 0 );
		currentTransform(1,3) = x( 1 );
		currentTransform(2,3) = x( 2 );


		qx = delta_x( 3 );
		qy = delta_x( 4 );
		qz = delta_x( 5 );
		qw = sqrt(1.0-qx*qx-qy*qy-qz*qz);

		Eigen::Matrix4d deltaTransform = Eigen::Matrix4d::Identity();
		deltaTransform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
		deltaTransform(0,3) = delta_x( 0 );
		deltaTransform(1,3) = delta_x( 1 );
		deltaTransform(2,3) = delta_x( 2 );

		Eigen::Matrix4d newTransform = deltaTransform * currentTransform;

		Eigen::Matrix< double, 6, 1 > x_new;
		x_new( 0 ) = newTransform(0,3);
		x_new( 1 ) = newTransform(1,3);
		x_new( 2 ) = newTransform(2,3);

		Eigen::Quaterniond q_new( newTransform.block<3,3>(0,0) );
		x_new( 3 ) = q_new.x();
		x_new( 4 ) = q_new.y();
		x_new( 5 ) = q_new.z();


//		FeatureAssociationList FALcopy;
//		FALcopy.reserve( featureAssociations_.size() );
//		for (MRCSRFAL::iterator it = featureAssociations_.begin(); it != featureAssociations_.end(); ++it)
//		{
//			MultiResolutionColorSurfelRegistration::FeatureAssociation assoc( it->src_idx_, it->dst_idx_ );
//
//			if (it->match == 0) {
//				assoc.landmark_pos = it->landmark_pos;
//				FALcopy.push_back(assoc);
//				continue;
//			}
//
//			Eigen::Vector3d deltaLM = (it->Hll + mu * Eigen::Matrix3d::Identity()).inverse() * (-it->bl - it->Hpl * deltaS);
//
//			assoc.landmark_pos = phi(phiInv(it->landmark_pos) + deltaLM);
//
//			FALcopy.push_back(assoc);
//		}

		for (MRCSRFAL::iterator it = featureAssociations_.begin(); it != featureAssociations_.end(); ++it)
		{

			it->tmp_landmark_pos = it->landmark_pos;

			if (it->match == 0) {
				continue;
			}

			Eigen::Vector3d deltaLM = (it->Hll + mu * Eigen::Matrix3d::Identity()).inverse() * (-it->bl - it->Hpl * deltaS);

			it->landmark_pos = phi(phiInv(it->landmark_pos) + deltaLM);

		}

		stopwatch.reset();

		new_error = 0.0;
		bool retVal2 = registrationErrorFunctionLM(x_new, new_error, surfelAssociations, featureAssociations_, featureAssocMahalDist);
		if (!retVal2)
		{
			std::cout << "2nd ErrorFunction for AreNo and Surfel failed\n";
			return false;
		}

		if( params_.debugFeatures_ )
			std::cout << "feature error function eval took: " << stopwatch.getTime() << "\n";

		double rho = (last_error - new_error) / (delta_x.transpose() * (mu * delta_x + poseOFdf));

		if (rho > 0) {

			x = x_new;

//			MRCSRFAL::iterator it2 = FALcopy.begin();
//			for (MRCSRFAL::iterator it = featureAssociations_.begin(); it != featureAssociations_.end(); ++it) {
//				it->landmark_pos = it2->landmark_pos;
//				it2++;
//			}

			mu *= std::max(0.333, 1.0 - pow(2.0 * rho - 1.0, 3.0));
			nu = 2;

			reevaluateGradient = true;

		} else {

			mu *= nu;
			nu *= 2.0;

			for (MRCSRFAL::iterator it = featureAssociations_.begin(); it != featureAssociations_.end(); ++it) {
				it->landmark_pos = it->tmp_landmark_pos;
			}
		}

		qx = x(3);
		qy = x(4);
		qz = x(5);
		qw = lastWSign_ * sqrt(1.0 - qx * qx - qy * qy - qz * qz);

		if (isnan(qw) || fabsf(qx) > 1.f || fabsf(qy) > 1.f || fabsf(qz) > 1.f) {
			return false;
		}

		transform.setIdentity();
		transform.block<3, 3>(0, 0) = Eigen::Matrix3d(
				Eigen::Quaterniond(qw, qx, qy, qz));
		transform(0, 3) = x(0);
		transform(1, 3) = x(1);
		transform(2, 3) = x(2);

		iter++;
	}
	return true;
}



bool MultiResolutionColorSurfelRegistration::estimateTransformation( MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution, pcl::PointCloud< pcl::PointXYZRGB >::Ptr correspondencesSourcePoints, pcl::PointCloud< pcl::PointXYZRGB >::Ptr correspondencesTargetPoints, int gradientIterations, int coarseToFineIterations, int fineIterations ) {

	params_.startResolution_ = startResolution;
	params_.stopResolution_ = stopResolution;

	source_ = &source;
	target_ = &target;

	correspondences_source_points_ = correspondencesSourcePoints;
	correspondences_target_points_ = correspondencesTargetPoints;

	// estimate transformation from maps
	target.clearAssociations();

	targetSamplingMap_ = algorithm::downsampleVectorOcTree(*target.octree_, false, target.octree_->max_depth_);

	bool retVal = true;
	if( params_.registerFeatures_ ) {

		pcl::StopWatch stopwatch;
		stopwatch.reset();
		associatePointFeatures();
		if( params_.debugFeatures_ )
			std::cout << "pf association took: " << stopwatch.getTime() << "\n";

		double mu = -1.0;
		double nu = 2.0;
		bool retVal = estimateTransformationLevenbergMarquardtPF( transform, gradientIterations + coarseToFineIterations + fineIterations, params_.pointFeatureMatchingCoarseImagePosMahalDist_, 1e-2, mu, nu );
		if( retVal ) {
			retVal = estimateTransformationLevenbergMarquardtPF( transform, gradientIterations + coarseToFineIterations + fineIterations, params_.pointFeatureMatchingFineImagePosMahalDist_, 1e-4, mu, nu );
		}

		if( !retVal )
			std::cout << "registration failed\n";

	}
	else {

		if( gradientIterations > 0 )
			retVal = estimateTransformationLevenbergMarquardt( transform, gradientIterations );

		if( !retVal )
			std::cout << "levenberg marquardt failed\n";

		Eigen::Matrix4d transformGradient = transform;

		if( retVal ) {

			bool retVal2 = estimateTransformationNewton( transform, coarseToFineIterations, fineIterations );
			if( !retVal2 ) {
				std::cout << "newton failed\n";
				transform = transformGradient;

				if( gradientIterations == 0 )
					retVal = false;
			}

		}

	}

	return retVal;

}


class MatchLogLikelihoodFunctor {
public:
	MatchLogLikelihoodFunctor( MultiResolutionColorSurfelRegistration::NodeLogLikelihoodList* nodes, MultiResolutionColorSurfelRegistration::Params params, MultiResolutionColorSurfelMap* source, MultiResolutionColorSurfelMap* target, const Eigen::Matrix4d& transform ) {
		nodes_ = nodes;
		params_ = params;
		source_ = source;
		target_ = target;
		transform_ = transform;

		normalStd = 0.125*M_PI;
		normalMinLogLikelihood = -0.5 * log( 2.0 * M_PI * normalStd ) - 8.0;

		targetToSourceTransform = transform_;
		currentRotation = Eigen::Matrix3d( targetToSourceTransform.block<3,3>(0,0) );
		currentRotationT = currentRotation.transpose();
		currentTranslation = Eigen::Vector3d( targetToSourceTransform.block<3,1>(0,3) );
	}

	~MatchLogLikelihoodFunctor() {}


	void operator()( const tbb::blocked_range<size_t>& r ) const {
		for( size_t i=r.begin(); i!=r.end(); ++i )
			(*this)((*nodes_)[i]);
	}


	void operator()( MultiResolutionColorSurfelRegistration::NodeLogLikelihood& node ) const {

		spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n = node.n_;

		double sumLogLikelihood = 0.0;

		const float processResolution = n->resolution();

		Eigen::Vector4d npos = n->getPosition().cast<double>();
		npos(3) = 1;
		Eigen::Vector4d npos_match_src = targetToSourceTransform * npos;

		// for match log likelihood: query in volume to check the neighborhood for the best matching (discretization issues)
		std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* > nodes;
		nodes.reserve(50);
		const double searchRadius = 2.0 * n->resolution();
		Eigen::Vector4f minPosition, maxPosition;
		minPosition[0] = npos_match_src(0) - searchRadius;
		minPosition[1] = npos_match_src(1) - searchRadius;
		minPosition[2] = npos_match_src(2) - searchRadius;
		maxPosition[0] = npos_match_src(0) + searchRadius;
		maxPosition[1] = npos_match_src(1) + searchRadius;
		maxPosition[2] = npos_match_src(2) + searchRadius;
		source_->octree_->getAllNodesInVolumeOnDepth( nodes, minPosition, maxPosition, n->depth_, false );

		Eigen::Matrix3d cov_add;
		cov_add.setZero();
		if( params_.add_smooth_pos_covariance_ ) {
			cov_add.setIdentity();
			cov_add *= params_.smooth_surface_cov_factor_ * processResolution*processResolution;
		}


		// only consider model surfels that are visible from the scene viewpoint under the given transformation

		for( unsigned int i = 0; i < MAX_NUM_SURFELS; i++ ) {

			MultiResolutionColorSurfelMap::Surfel* modelSurfel = &n->value_.surfels_[i];

			if( modelSurfel->num_points_ < MIN_SURFEL_POINTS ) {
				continue;
			}

			// transform surfel mean with current transform and find corresponding node in source for current resolution
			// find corresponding surfel in node via the transformed view direction of the surfel

			Eigen::Vector4d pos;
			pos.block<3,1>(0,0) = modelSurfel->mean_.block<3,1>(0,0);
			pos(3,0) = 1.f;

			Eigen::Vector4d dir;
			dir.block<3,1>(0,0) = modelSurfel->initial_view_dir_;
			dir(3,0) = 0.f; // pure rotation

			Eigen::Vector4d pos_match_src = targetToSourceTransform * pos;
			Eigen::Vector4d dir_match_src = targetToSourceTransform * dir;

			// precalculate log likelihood when surfel is not matched in the scene
			Eigen::Matrix3d cov2 = modelSurfel->cov_.block<3,3>(0,0);
			cov2 += cov_add;

			Eigen::Matrix3d cov2_RT = cov2 * currentRotationT;
			Eigen::Matrix3d cov2_rotated = (currentRotation * cov2_RT).eval();

			double nomatch_loglikelihood = -0.5 * log( 8.0 * M_PI * M_PI * M_PI * cov2_rotated.determinant() ) - 24.0;

			if( params_.match_likelihood_use_color_ ) {
				nomatch_loglikelihood += -0.5 * log( 8.0 * M_PI * M_PI * M_PI * modelSurfel->cov_.block<3,3>(3,3).determinant() ) - 24.0;
			}

			nomatch_loglikelihood += normalMinLogLikelihood;

			if( std::isinf<double>(nomatch_loglikelihood) || std::isnan<double>(nomatch_loglikelihood) )
				continue;

			double bestSurfelLogLikelihood = nomatch_loglikelihood;

//			// is model surfel visible from the scene viewpoint?
//			// assumption: scene viewpoint in (0,0,0)
//			if( dir_match_src.block<3,1>(0,0).dot( pos_match_src.block<3,1>(0,0) / pos_match_src.block<3,1>(0,0).norm() ) < cos(0.25*M_PI) ) {
//				sumLogLikelihood += bestSurfelLogLikelihood;
//				continue;
//			}

			for( std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* >::iterator it = nodes.begin(); it != nodes.end(); ++it ) {

				spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_src = *it;

				// find best matching surfel for the view direction in the scene map
				MultiResolutionColorSurfelMap::Surfel* bestMatchSurfel = NULL;
				double bestMatchDist = -1.f;
				for( unsigned int k = 0; k < MAX_NUM_SURFELS; k++ ) {

					const double dist = dir_match_src.block<3,1>(0,0).dot( n_src->value_.surfels_[k].initial_view_dir_ );
					if( dist > bestMatchDist ) {
						bestMatchSurfel = &n_src->value_.surfels_[k];
						bestMatchDist = dist;
					}
				}


				// do only associate on the same resolution
				// no match? use maximum distance log likelihood for this surfel
				if( bestMatchSurfel->num_points_ < MIN_SURFEL_POINTS ) {
					continue;
				}


				Eigen::Vector3d diff_pos = bestMatchSurfel->mean_.block<3,1>(0,0) - pos_match_src.block<3,1>(0,0);


				Eigen::Matrix3d cov1 = bestMatchSurfel->cov_.block<3,3>(0,0);
				cov1 += cov_add;

//				// apply curvature dependent covariance
//				cov1.block<3,3>(0,0) += bestMatchSurfel->cov_add_;

				Eigen::Matrix3d cov = cov1 + cov2_rotated;
				Eigen::Matrix3d invcov = cov.inverse().eval();

				double exponent = -0.5 * diff_pos.dot(invcov * diff_pos);
				exponent = std::max( -12.0, exponent ); // -32: -0.5 * ( 16 + 16 + 16 + 16 )
				double loglikelihood = -0.5 * log( 8.0 * M_PI * M_PI * M_PI * cov.determinant() ) + exponent;



				Eigen::Vector3d diff_col = bestMatchSurfel->mean_.block<3,1>(3,0) - modelSurfel->mean_.block<3,1>(3,0);
				if( fabs(diff_col(0)) < params_.luminance_damp_diff_ )
					diff_col(0) = 0;
				if( fabs(diff_col(1)) < params_.color_damp_diff_ )
					diff_col(1) = 0;
				if( fabs(diff_col(2)) < params_.color_damp_diff_ )
					diff_col(2) = 0;

				if( diff_col(0) < 0 )
					diff_col(0) += params_.luminance_damp_diff_;
				if( diff_col(1) < 0 )
					diff_col(1) += params_.color_damp_diff_;
				if( diff_col(2) < 0 )
					diff_col(2) += params_.color_damp_diff_;

				if( diff_col(0) > 0 )
					diff_col(0) -= params_.luminance_damp_diff_;
				if( diff_col(1) > 0 )
					diff_col(1) -= params_.color_damp_diff_;
				if( diff_col(2) > 0 )
					diff_col(2) -= params_.color_damp_diff_;

				if( params_.match_likelihood_use_color_ ) {
					const Eigen::Matrix3d cov_cc = bestMatchSurfel->cov_.block<3,3>(3,3) + modelSurfel->cov_.block<3,3>(3,3) + 0.01 * Eigen::Matrix3d::Identity();
					double color_exponent = std::max( -12.0, - 0.5 * diff_col.dot( (cov_cc.inverse() * diff_col ) ) );
					double color_loglikelihood = -0.5 * log( 8.0 * M_PI * M_PI * M_PI * cov_cc.determinant() ) + exponent;
					loglikelihood += color_loglikelihood;
				}

				// test: also consider normal orientation in the likelihood!!
				Eigen::Vector4d normal_src;
				normal_src.block<3,1>(0,0) = modelSurfel->normal_;
				normal_src(3,0) = 0.0;
				normal_src = (targetToSourceTransform * normal_src).eval();

				double normalError = std::min( 2.0 * normalStd, acos( normal_src.block<3,1>(0,0).dot( bestMatchSurfel->normal_ ) ) );
				double normalExponent = -0.5 * normalError * normalError / ( normalStd*normalStd );
				double normalLogLikelihood = -0.5 * log( 2.0 * M_PI * normalStd ) + normalExponent;


				if( std::isinf<double>(nomatch_loglikelihood) || std::isnan<double>( exponent ) ) {
					continue;
				}
				if( std::isinf<double>(nomatch_loglikelihood) || std::isnan<double>(loglikelihood) )
					continue;

				bestSurfelLogLikelihood = std::max( bestSurfelLogLikelihood, loglikelihood + normalLogLikelihood );

			}

			sumLogLikelihood += bestSurfelLogLikelihood;
		}

		node.loglikelihood_ = sumLogLikelihood;

	}


	MultiResolutionColorSurfelRegistration::NodeLogLikelihoodList* nodes_;
	MultiResolutionColorSurfelRegistration::Params params_;
	MultiResolutionColorSurfelMap* source_;
	MultiResolutionColorSurfelMap* target_;
	Eigen::Matrix4d transform_;

	double normalStd;
	double normalMinLogLikelihood;


	Eigen::Matrix4d targetToSourceTransform;
	Eigen::Matrix3d currentRotation;
	Eigen::Matrix3d currentRotationT;
	Eigen::Vector3d currentTranslation;

};

// transform: transforms source to target
// intended to have the "smaller" map (the model) in target
double MultiResolutionColorSurfelRegistration::matchLogLikelihood( MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, Eigen::Matrix4d& transform ) {

	targetSamplingMap_ = algorithm::downsampleVectorOcTree(*target.octree_, false, target.octree_->max_depth_);

	int maxDepth = target.octree_->max_depth_;

	int countNodes = 0;
	for( int d = maxDepth; d >= 0; d-- ) {
		countNodes += targetSamplingMap_[d].size();
	}

	NodeLogLikelihoodList nodes;
	nodes.reserve( countNodes );

	for( int d = maxDepth; d >= 0; d-- ) {

		for( unsigned int i = 0; i < targetSamplingMap_[d].size(); i++ )
			nodes.push_back( NodeLogLikelihood( targetSamplingMap_[d][i] ) );

	}


	MatchLogLikelihoodFunctor mlf( &nodes, params_, &source, &target, transform );

	if( params_.parallel_ )
		tbb::parallel_for_each( nodes.begin(), nodes.end(), mlf );
	else
		std::for_each( nodes.begin(), nodes.end(), mlf );

	double sumLogLikelihood = 0.0;
	for( unsigned int i = 0; i < nodes.size(); i++ ) {
		sumLogLikelihood += nodes[i].loglikelihood_;
	}

	return sumLogLikelihood;

//
//	double sumLogLikelihood = 0.0;
//
//	const double normalStd = 0.125*M_PI;
//	const double normalMinLogLikelihood = -0.5 * log( 2.0 * M_PI * normalStd ) - 8.0;
//
//	Eigen::Matrix4d targetToSourceTransform = transform;
//	Eigen::Matrix3d currentRotation = Eigen::Matrix3d( targetToSourceTransform.block<3,3>(0,0) );
//	Eigen::Matrix3d currentRotationT = currentRotation.transpose();
//	Eigen::Vector3d currentTranslation = Eigen::Vector3d( targetToSourceTransform.block<3,1>(0,3) );
//
//	// start at highest resolution in the tree and compare recursively
//	MultiResolutionColorSurfelRegistration::SurfelAssociationList surfelAssociations;
//	std::list< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* > openNodes;
//	openNodes.push_back( target.octree_->root_ );
//	while( !openNodes.empty() ) {
//		spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n = openNodes.front();
//		openNodes.pop_front();
//
//		for( unsigned int i = 0; i < 8; i++ ) {
//			if( n->children_[i] )
//				openNodes.push_back( n->children_[i] );
//		}
//
//		const float processResolution = n->resolution();
//
//		Eigen::Vector4d npos = n->getPosition().cast<double>();
//		Eigen::Vector4d npos_match_src = targetToSourceTransform * npos;
//
//		// for match log likelihood: query in volume to check the neighborhood for the best matching (discretization issues)
//		std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* > nodes;
//		nodes.reserve(50);
//		const double searchRadius = 2.0 * n->resolution();
//		Eigen::Vector4f minPosition, maxPosition;
//		minPosition[0] = npos_match_src(0) - searchRadius;
//		minPosition[1] = npos_match_src(1) - searchRadius;
//		minPosition[2] = npos_match_src(2) - searchRadius;
//		maxPosition[0] = npos_match_src(0) + searchRadius;
//		maxPosition[1] = npos_match_src(1) + searchRadius;
//		maxPosition[2] = npos_match_src(2) + searchRadius;
//		source.octree_->getAllNodesInVolumeOnDepth( nodes, minPosition, maxPosition, n->depth_, false );
//
//		Eigen::Matrix3d cov_add;
//		cov_add.setZero();
//		if( params_.add_smooth_pos_covariance_ ) {
//			cov_add.setIdentity();
//			cov_add *= params_.smooth_surface_cov_factor_ * processResolution*processResolution;
//		}
//
//
//		// only consider model surfels that are visible from the scene viewpoint under the given transformation
//
//		for( unsigned int i = 0; i < MAX_NUM_SURFELS; i++ ) {
//
//			MultiResolutionColorSurfelMap::Surfel* modelSurfel = &n->value_.surfels_[i];
//
//			if( modelSurfel->num_points_ < MIN_SURFEL_POINTS ) {
//				continue;
//			}
//
//			// transform surfel mean with current transform and find corresponding node in source for current resolution
//			// find corresponding surfel in node via the transformed view direction of the surfel
//
//			Eigen::Vector4d pos;
//			pos.block<3,1>(0,0) = modelSurfel->mean_.block<3,1>(0,0);
//			pos(3,0) = 1.f;
//
//			Eigen::Vector4d dir;
//			dir.block<3,1>(0,0) = modelSurfel->initial_view_dir_;
//			dir(3,0) = 0.f; // pure rotation
//
//			Eigen::Vector4d pos_match_src = targetToSourceTransform * pos;
//			Eigen::Vector4d dir_match_src = targetToSourceTransform * dir;
//
//			// precalculate log likelihood when surfel is not matched in the scene
//			Eigen::Matrix3d cov2 = modelSurfel->cov_.block<3,3>(0,0);
//			cov2 += cov_add;
//
//			Eigen::Matrix3d cov2_RT = cov2 * currentRotationT;
//			Eigen::Matrix3d cov2_rotated = (currentRotation * cov2_RT).eval();
//
//			double nomatch_loglikelihood = -0.5 * log( 8.0 * M_PI * M_PI * M_PI * cov2_rotated.determinant() ) - 24.0;
//
//			if( params_.match_likelihood_use_color_ ) {
//				nomatch_loglikelihood += -0.5 * log( 8.0 * M_PI * M_PI * M_PI * modelSurfel->cov_.block<3,3>(3,3).determinant() ) - 24.0;
//			}
//
//			nomatch_loglikelihood += normalMinLogLikelihood;
//
//			if( std::isinf<double>(nomatch_loglikelihood) || std::isnan<double>(nomatch_loglikelihood) )
//				continue;
//
//			double bestSurfelLogLikelihood = nomatch_loglikelihood;
//
//			// is model surfel visible from the scene viewpoint?
//			// assumption: scene viewpoint in (0,0,0)
//			if( dir_match_src.block<3,1>(0,0).dot( pos_match_src.block<3,1>(0,0) / pos_match_src.block<3,1>(0,0).norm() ) < cos(0.25*M_PI) ) {
//				sumLogLikelihood += bestSurfelLogLikelihood;
//				continue;
//			}
//
//			for( std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* >::iterator it = nodes.begin(); it != nodes.end(); ++it ) {
//
//				spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_src = *it;
//
//				// find best matching surfel for the view direction in the scene map
//				MultiResolutionColorSurfelMap::Surfel* bestMatchSurfel = NULL;
//				double bestMatchDist = -1.f;
//				for( unsigned int k = 0; k < MAX_NUM_SURFELS; k++ ) {
//
//					const double dist = dir_match_src.block<3,1>(0,0).dot( n_src->value_.surfels_[k].initial_view_dir_ );
//					if( dist > bestMatchDist ) {
//						bestMatchSurfel = &n_src->value_.surfels_[k];
//						bestMatchDist = dist;
//					}
//				}
//
//
//				// do only associate on the same resolution
//				// no match? use maximum distance log likelihood for this surfel
//				if( bestMatchSurfel->num_points_ < MIN_SURFEL_POINTS ) {
//					continue;
//				}
//
//
//				Eigen::Vector3d diff_pos = bestMatchSurfel->mean_.block<3,1>(0,0) - pos_match_src.block<3,1>(0,0);
//
//
//				Eigen::Matrix3d cov1 = bestMatchSurfel->cov_.block<3,3>(0,0);
//				cov1 += cov_add;
//
////				// apply curvature dependent covariance
////				cov1.block<3,3>(0,0) += bestMatchSurfel->cov_add_;
//
//				Eigen::Matrix3d cov = cov1 + cov2_rotated;
//				Eigen::Matrix3d invcov = cov.inverse().eval();
//
//				double exponent = -0.5 * diff_pos.dot(invcov * diff_pos);
//				exponent = std::max( -12.0, exponent ); // -32: -0.5 * ( 16 + 16 + 16 + 16 )
//				double loglikelihood = -0.5 * log( 8.0 * M_PI * M_PI * M_PI * cov.determinant() ) + exponent;
//
//
//
//				Eigen::Vector3d diff_col = bestMatchSurfel->mean_.block<3,1>(3,0) - modelSurfel->mean_.block<3,1>(3,0);
//				if( fabs(diff_col(0)) < params_.luminance_damp_diff_ )
//					diff_col(0) = 0;
//				if( fabs(diff_col(1)) < params_.color_damp_diff_ )
//					diff_col(1) = 0;
//				if( fabs(diff_col(2)) < params_.color_damp_diff_ )
//					diff_col(2) = 0;
//
//				if( diff_col(0) < 0 )
//					diff_col(0) += params_.luminance_damp_diff_;
//				if( diff_col(1) < 0 )
//					diff_col(1) += params_.color_damp_diff_;
//				if( diff_col(2) < 0 )
//					diff_col(2) += params_.color_damp_diff_;
//
//				if( diff_col(0) > 0 )
//					diff_col(0) -= params_.luminance_damp_diff_;
//				if( diff_col(1) > 0 )
//					diff_col(1) -= params_.color_damp_diff_;
//				if( diff_col(2) > 0 )
//					diff_col(2) -= params_.color_damp_diff_;
//
//				if( params_.match_likelihood_use_color_ ) {
//					const Eigen::Matrix3d cov_cc = bestMatchSurfel->cov_.block<3,3>(3,3) + modelSurfel->cov_.block<3,3>(3,3) + 0.01 * Eigen::Matrix3d::Identity();
//					double color_exponent = std::max( -12.0, - 0.5 * diff_col.dot( (cov_cc.inverse() * diff_col ) ) );
//					double color_loglikelihood = -0.5 * log( 8.0 * M_PI * M_PI * M_PI * cov_cc.determinant() ) + exponent;
//					loglikelihood += color_loglikelihood;
//				}
//
//				// test: also consider normal orientation in the likelihood!!
//				Eigen::Vector4d normal_src;
//				normal_src.block<3,1>(0,0) = modelSurfel->normal_;
//				normal_src(3,0) = 0.0;
//				normal_src = (targetToSourceTransform * normal_src).eval();
//
//				double normalError = std::min( 2.0 * normalStd, acos( normal_src.block<3,1>(0,0).dot( bestMatchSurfel->normal_ ) ) );
//				double normalExponent = -0.5 * normalError * normalError / ( normalStd*normalStd );
//				double normalLogLikelihood = -0.5 * log( 2.0 * M_PI * normalStd ) + normalExponent;
//
//
//				if( std::isinf<double>(nomatch_loglikelihood) || std::isnan<double>( exponent ) ) {
//					continue;
//				}
//				if( std::isinf<double>(nomatch_loglikelihood) || std::isnan<double>(loglikelihood) )
//					continue;
//
//				bestSurfelLogLikelihood = std::max( bestSurfelLogLikelihood, loglikelihood + normalLogLikelihood );
//
//			}
//
//			sumLogLikelihood += bestSurfelLogLikelihood;
//		}
//
//
//	}
//
//	return sumLogLikelihood;


}


class SelfMatchLogLikelihoodFunctor {
public:
	SelfMatchLogLikelihoodFunctor( MultiResolutionColorSurfelRegistration::NodeLogLikelihoodList* nodes, MultiResolutionColorSurfelRegistration::Params params, MultiResolutionColorSurfelMap* target ) {
		nodes_ = nodes;
		params_ = params;
		target_ = target;

		normalStd = 0.125*M_PI;
		normalMinLogLikelihood = -0.5 * log( 2.0 * M_PI * normalStd ) - 8.0;
	}

	~SelfMatchLogLikelihoodFunctor() {}


	void operator()( const tbb::blocked_range<size_t>& r ) const {
		for( size_t i=r.begin(); i!=r.end(); ++i )
			(*this)((*nodes_)[i]);
	}


	void operator()( MultiResolutionColorSurfelRegistration::NodeLogLikelihood& node ) const {

		spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n = node.n_;

		double sumLogLikelihood = 0.0;

		const float processResolution = n->resolution();

		Eigen::Matrix3d cov_add;
		cov_add.setZero();
		if( params_.add_smooth_pos_covariance_ ) {
			cov_add.setIdentity();
			cov_add *= params_.smooth_surface_cov_factor_ * processResolution*processResolution;
		}


		// only consider model surfels that are visible from the scene viewpoint under the given transformation

		for( unsigned int i = 0; i < MAX_NUM_SURFELS; i++ ) {

			MultiResolutionColorSurfelMap::Surfel* modelSurfel = &n->value_.surfels_[i];

			if( modelSurfel->num_points_ < MIN_SURFEL_POINTS ) {
				continue;
			}

			// precalculate log likelihood when surfel is not matched in the scene
			Eigen::Matrix3d cov2 = modelSurfel->cov_.block<3,3>(0,0);
			cov2 += cov_add;

			double nomatch_loglikelihood = -0.5 * log( 8.0 * M_PI * M_PI * M_PI * cov2.determinant() ) - 24.0;

			if( params_.match_likelihood_use_color_ ) {
				nomatch_loglikelihood += -0.5 * log( 8.0 * M_PI * M_PI * M_PI * modelSurfel->cov_.block<3,3>(3,3).determinant() ) - 24.0;
			}

			nomatch_loglikelihood += normalMinLogLikelihood;

			if( std::isinf<double>(nomatch_loglikelihood) || std::isnan<double>(nomatch_loglikelihood) )
				continue;

			double bestSurfelLogLikelihood = nomatch_loglikelihood;

			Eigen::Matrix3d cov = 2.0*cov2;
//			Eigen::Matrix3d invcov = cov.inverse().eval();

			double loglikelihood = -0.5 * log( 8.0 * M_PI * M_PI * M_PI * cov.determinant() );

			if( params_.match_likelihood_use_color_ ) {
				const Eigen::Matrix3d cov_cc = 2.0 * modelSurfel->cov_.block<3,3>(3,3) + 0.01 * Eigen::Matrix3d::Identity();
				double color_loglikelihood = -0.5 * log( 8.0 * M_PI * M_PI * M_PI * cov_cc.determinant() );
				loglikelihood += color_loglikelihood;
			}

			double normalLogLikelihood = -0.5 * log( 2.0 * M_PI * normalStd );


			if( std::isinf<double>(nomatch_loglikelihood) || std::isnan<double>(loglikelihood) )
				continue;

			bestSurfelLogLikelihood = std::max( bestSurfelLogLikelihood, loglikelihood + normalLogLikelihood );


			sumLogLikelihood += bestSurfelLogLikelihood;

		}

		node.loglikelihood_ = sumLogLikelihood;

	}


	MultiResolutionColorSurfelRegistration::NodeLogLikelihoodList* nodes_;
	MultiResolutionColorSurfelRegistration::Params params_;
	MultiResolutionColorSurfelMap* target_;

	double normalStd;
	double normalMinLogLikelihood;

};


double MultiResolutionColorSurfelRegistration::selfMatchLogLikelihood( MultiResolutionColorSurfelMap& target ) {

	targetSamplingMap_ = algorithm::downsampleVectorOcTree(*target.octree_, false, target.octree_->max_depth_);

	int maxDepth = target.octree_->max_depth_;

	int countNodes = 0;
	for( int d = maxDepth; d >= 0; d-- ) {
		countNodes += targetSamplingMap_[d].size();
	}

	NodeLogLikelihoodList nodes;
	nodes.reserve( countNodes );

	for( int d = maxDepth; d >= 0; d-- ) {

		for( unsigned int i = 0; i < targetSamplingMap_[d].size(); i++ )
			nodes.push_back( NodeLogLikelihood( targetSamplingMap_[d][i] ) );

	}


	SelfMatchLogLikelihoodFunctor mlf( &nodes, params_, &target );

	if( params_.parallel_ )
		tbb::parallel_for_each( nodes.begin(), nodes.end(), mlf );
	else
		std::for_each( nodes.begin(), nodes.end(), mlf );

	double sumLogLikelihood = 0.0;
	for( unsigned int i = 0; i < nodes.size(); i++ ) {
		sumLogLikelihood += nodes[i].loglikelihood_;
	}

	return sumLogLikelihood;

}


bool MultiResolutionColorSurfelRegistration::estimatePoseCovariance( Eigen::Matrix< double, 6, 6 >& poseCov, MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution ) {

	target.clearAssociations();

	float minResolution = std::min( startResolution, stopResolution );
	float maxResolution = std::max( startResolution, stopResolution );

	algorithm::OcTreeSamplingVectorMap<float, MultiResolutionColorSurfelMap::NodeValue> targetSamplingMap = algorithm::downsampleVectorOcTree(*target.octree_, false, target.octree_->max_depth_);

	double sumWeight = 0.0;

	Eigen::Quaterniond q( transform.block<3,3>(0,0) );

	const double tx = transform(0,3);
	const double ty = transform(1,3);
	const double tz = transform(2,3);
	const double qx = q.x();
	const double qy = q.y();
	const double qz = q.z();
	const double qw = q.w();


	MultiResolutionColorSurfelRegistration::SurfelAssociationList surfelAssociations;
	associateMapsBreadthFirstParallel( surfelAssociations, source, target, targetSamplingMap, transform, 0.99f*minResolution, 1.01f*maxResolution, 2.f, 2.f*maxResolution, false );


	GradientFunctor gf( &surfelAssociations, params_, tx, ty, tz, qx, qy, qz, qw, false, true, true, true );

	if( params_.parallel_ )
		tbb::parallel_for( tbb::blocked_range<size_t>(0,surfelAssociations.size()), gf );
	else
		std::for_each( surfelAssociations.begin(), surfelAssociations.end(), gf );

	Eigen::Matrix< double, 6, 6 > d2f, JSzJ;
	d2f.setZero();
	JSzJ.setZero();

	for( MultiResolutionColorSurfelRegistration::SurfelAssociationList::iterator it = surfelAssociations.begin(); it != surfelAssociations.end(); ++it ) {

		if( !it->match )
			continue;

//		d2f += it->d2f;
//		JSzJ += it->weight * it->JSzJ;
		d2f += it->weight * it->d2f;
		JSzJ += it->weight * it->weight * it->JSzJ;
		sumWeight += it->weight;

	}


	if( sumWeight <= 1e-10 ) {
		poseCov.setIdentity();
		return false;
	}
	else if( sumWeight < params_.registration_min_num_surfels_ ) {
		std::cout << "not enough surfels for robust matching\n";
		poseCov.setIdentity();
		return false;
	}
	else {
		d2f /= sumWeight;
		JSzJ /= sumWeight * sumWeight;
	}

	poseCov.setZero();

	if( fabsf(d2f.determinant()) < 1e-8 ) {
		poseCov.setIdentity();
		return false;
	}

	poseCov = d2f.inverse() * JSzJ * d2f.inverse();


	return true;


}


bool MultiResolutionColorSurfelRegistration::estimatePoseCovarianceLM( Eigen::Matrix< double, 6, 6 >& poseCov, MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution ) {

	target.clearAssociations();

	float minResolution = std::min( startResolution, stopResolution );
	float maxResolution = std::max( startResolution, stopResolution );

	algorithm::OcTreeSamplingVectorMap<float, MultiResolutionColorSurfelMap::NodeValue> targetSamplingMap = algorithm::downsampleVectorOcTree(*target.octree_, false, target.octree_->max_depth_);

	double sumWeight = 0.0;

	Eigen::Quaterniond q( transform.block<3,3>(0,0) );

	const double tx = transform(0,3);
	const double ty = transform(1,3);
	const double tz = transform(2,3);
	const double qx = q.x();
	const double qy = q.y();
	const double qz = q.z();
	const double qw = q.w();


	MultiResolutionColorSurfelRegistration::SurfelAssociationList surfelAssociations;
	associateMapsBreadthFirstParallel( surfelAssociations, source, target, targetSamplingMap, transform, 0.99f*minResolution, 1.01f*maxResolution, 2.f, 2.f*maxResolution, false );


	GradientFunctorLM gf( &surfelAssociations, params_, tx, ty, tz, qx, qy, qz, qw, true, true );

	if( params_.parallel_ )
		tbb::parallel_for( tbb::blocked_range<size_t>(0,surfelAssociations.size()), gf );
	else
		std::for_each( surfelAssociations.begin(), surfelAssociations.end(), gf );

	Eigen::Matrix< double, 6, 6 > d2f, JSzJ;
	d2f.setZero();
	JSzJ.setZero();

	for( MultiResolutionColorSurfelRegistration::SurfelAssociationList::iterator it = surfelAssociations.begin(); it != surfelAssociations.end(); ++it ) {

		if( !it->match )
			continue;

		d2f += it->weight * it->dh_dx.transpose() * it->W * it->dh_dx;
		JSzJ += it->weight * it->weight * it->JSzJ;
		sumWeight += it->weight;

	}


	if( sumWeight <= 1e-10 ) {
		poseCov.setIdentity();
		return false;
	}
	else if( sumWeight < params_.registration_min_num_surfels_ ) {
		std::cout << "not enough surfels for robust matching\n";
		poseCov.setIdentity();
		return false;
	}
	else {
		d2f /= sumWeight;
		JSzJ /= sumWeight * sumWeight;
	}

	poseCov.setZero();

	if( fabsf(d2f.determinant()) < 1e-8 ) {
		poseCov.setIdentity();
		return false;
	}

	poseCov = d2f.inverse() * JSzJ * d2f.inverse();

	return true;


}


//MultiFrameMultiResolutionColorSurfelRegistration::MultiFrameMultiResolutionColorSurfelRegistration() {
//
//	start_resolution_ = 0.f;
//	stop_resolution_ = std::numeric_limits<float>::max();
//
//	// build up derivatives of rotation and translation for the transformation variables
//	dT_tx_.setZero();
//	dT_ty_.setZero();
//	dT_tz_.setZero();
//	dT_qx_.setZero();
//	dT_qy_.setZero();
//	dT_qz_.setZero();
//
//	dT_tx_(0,3) = 1.0;
//
//	dT_ty_(1,3) = 1.0;
//
//	dT_tz_(2,3) = 1.0;
//
//	dT_qx_(1,2) = -2.0;
//	dT_qx_(2,1) = +2.0;
//
//	dT_qy_(0,2) = +2.0;
//	dT_qy_(2,0) = -2.0;
//
//	dT_qz_(0,1) = -2.0;
//	dT_qz_(1,0) = +2.0;
//
//}
//
//
//
//
//void MultiFrameMultiResolutionColorSurfelRegistration::addFramePair( MultiResolutionColorSurfelMap* source, MultiResolutionColorSurfelMap* target, const Eigen::Matrix4d& transformGuess ) {
//
//	registration_pairs_.push_back( FramePair( source, target, transformGuess ) );
//
//}
//
//void MultiFrameMultiResolutionColorSurfelRegistration::addTargetPoseConstraint( MultiResolutionColorSurfelMap* target_from, MultiResolutionColorSurfelMap* target_to, const Eigen::Matrix4d& refFrameTransform, const Eigen::Matrix< double, 7, 1 >& prior_pose_mean, const Eigen::Matrix< double, 6, 1 >& prior_pose_variances ) {
//
//	int id_from = 0, id_to = 0;
//	for( unsigned int i = 0; i < registration_pairs_.size(); i++ ) {
//		if( registration_pairs_[i].target_ == target_from )
//			id_from = i;
//		if( registration_pairs_[i].target_ == target_to )
//			id_to = i;
//	}
//
//	target_pose_constraints_.push_back( FramePairConstraint( target_from, target_to, refFrameTransform, prior_pose_mean, prior_pose_variances ) );
//	target_pose_constraint_ids_.push_back( std::pair<int,int>( id_from, id_to ) );
//
//}
//
//
//Eigen::Matrix< double, 6, 1 > MultiFrameMultiResolutionColorSurfelRegistration::dpose_dT_times_dT_ddelta( const Eigen::Matrix4d& dT_ddelta ) {
//
//	Eigen::Matrix< double, 6, 1 > res = Eigen::Matrix< double, 6, 1 >::Zero();
//
//	res.block<3,1>(0,0) = dT_ddelta.block<3,1>(0,3);
//
//	Eigen::Matrix< double, 3, 9 > dq_dR;
//	g2o::internal::compute_dq_dR( dq_dR, dT_ddelta(0,0), dT_ddelta(1,0), dT_ddelta(2,0), dT_ddelta(0,1), dT_ddelta(1,1), dT_ddelta(2,1), dT_ddelta(0,2), dT_ddelta(1,2), dT_ddelta(2,2) );
//
//	Eigen::Matrix< double, 9, 1 > R;
//	R.block<3,1>(0,0) = dT_ddelta.block<3,1>(0,0);
//	R.block<3,1>(3,0) = dT_ddelta.block<3,1>(0,1);
//	R.block<3,1>(6,0) = dT_ddelta.block<3,1>(0,2);
//
//	res.block<3,1>(3,0) = dq_dR * R;
//
//	return res;
//
//}
//
//
//void MultiFrameMultiResolutionColorSurfelRegistration::poseConstraintError( Eigen::Matrix< double, 6, 1 >& poseDiff, const Eigen::Matrix4d& refFrameTransform, const Eigen::Matrix< double, 7, 1 >& poseConstraint, const Eigen::Matrix4d& poseEstimateFrom, const Eigen::Matrix4d& poseEstimateTo ) {
//
//	Eigen::Matrix4d poseConstraintTransform = Eigen::Matrix4d::Identity();
//	poseConstraintTransform.block<3,3>(0,0) = Eigen::Quaterniond( poseConstraint(6), poseConstraint(3), poseConstraint(4), poseConstraint(5) ).matrix();
//	poseConstraintTransform(0,3) = poseConstraint(0);
//	poseConstraintTransform(1,3) = poseConstraint(1);
//	poseConstraintTransform(2,3) = poseConstraint(2);
//
//	Eigen::Matrix4d diffTransform = poseConstraintTransform.inverse() * poseEstimateTo.inverse() * refFrameTransform * poseEstimateFrom;
//	Eigen::Quaterniond diffQ( diffTransform.block<3,3>(0,0) );
//	Eigen::Matrix< double, 6, 1 > d;
//	d(0) = diffTransform(0,3);
//	d(1) = diffTransform(1,3);
//	d(2) = diffTransform(2,3);
//	d(3) = diffQ.x();
//	d(4) = diffQ.y();
//	d(5) = diffQ.z();
//
//	poseDiff = d;
//
//}
//
//
//
//void MultiFrameMultiResolutionColorSurfelRegistration::poseConstraintErrorWithFirstDerivative( Eigen::Matrix< double, 6, 1 >& poseDiff, Eigen::Matrix< double, 6, 6 >& J_from, Eigen::Matrix< double, 6, 6 >& J_to, const Eigen::Matrix4d& refFrameTransform, const Eigen::Matrix< double, 7, 1 >& poseConstraint, const Eigen::Matrix4d& poseEstimateFrom, const Eigen::Matrix4d& poseEstimateTo ) {
//
//	Eigen::Matrix4d poseConstraintTransform = Eigen::Matrix4d::Identity();
//	poseConstraintTransform.block<3,3>(0,0) = Eigen::Quaterniond( poseConstraint(6), poseConstraint(3), poseConstraint(4), poseConstraint(5) ).matrix();
//	poseConstraintTransform(0,3) = poseConstraint(0);
//	poseConstraintTransform(1,3) = poseConstraint(1);
//	poseConstraintTransform(2,3) = poseConstraint(2);
//
//	Eigen::Matrix4d diffTransform = poseConstraintTransform.inverse() * poseEstimateTo.inverse() * refFrameTransform * poseEstimateFrom;
//	Eigen::Quaterniond diffQ( diffTransform.block<3,3>(0,0) );
//	Eigen::Matrix< double, 6, 1 > d;
//	d(0) = diffTransform(0,3);
//	d(1) = diffTransform(1,3);
//	d(2) = diffTransform(2,3);
//	d(3) = diffQ.x();
//	d(4) = diffQ.y();
//	d(5) = diffQ.z();
//
//	poseDiff = d;
//
//	// determine Jacobians
//	J_from.block<6,1>(0,0) = dpose_dT_times_dT_ddelta( poseConstraintTransform.inverse() * poseEstimateTo.inverse() * refFrameTransform * dT_tx_ * poseEstimateFrom );
//	J_from.block<6,1>(0,1) = dpose_dT_times_dT_ddelta( poseConstraintTransform.inverse() * poseEstimateTo.inverse() * refFrameTransform * dT_ty_ * poseEstimateFrom );
//	J_from.block<6,1>(0,2) = dpose_dT_times_dT_ddelta( poseConstraintTransform.inverse() * poseEstimateTo.inverse() * refFrameTransform * dT_tz_ * poseEstimateFrom );
//	J_from.block<6,1>(0,3) = dpose_dT_times_dT_ddelta( poseConstraintTransform.inverse() * poseEstimateTo.inverse() * refFrameTransform * dT_qx_ * poseEstimateFrom );
//	J_from.block<6,1>(0,4) = dpose_dT_times_dT_ddelta( poseConstraintTransform.inverse() * poseEstimateTo.inverse() * refFrameTransform * dT_qy_ * poseEstimateFrom );
//	J_from.block<6,1>(0,5) = dpose_dT_times_dT_ddelta( poseConstraintTransform.inverse() * poseEstimateTo.inverse() * refFrameTransform * dT_qz_ * poseEstimateFrom );
//
//	J_to.block<6,1>(0,0) = dpose_dT_times_dT_ddelta( poseConstraintTransform.inverse() * poseEstimateTo.inverse() * -dT_tx_ * refFrameTransform * poseEstimateFrom );
//	J_to.block<6,1>(0,1) = dpose_dT_times_dT_ddelta( poseConstraintTransform.inverse() * poseEstimateTo.inverse() * -dT_ty_ * refFrameTransform * poseEstimateFrom );
//	J_to.block<6,1>(0,2) = dpose_dT_times_dT_ddelta( poseConstraintTransform.inverse() * poseEstimateTo.inverse() * -dT_tz_ * refFrameTransform * poseEstimateFrom );
//	J_to.block<6,1>(0,3) = dpose_dT_times_dT_ddelta( poseConstraintTransform.inverse() * poseEstimateTo.inverse() * -dT_qx_ * refFrameTransform * poseEstimateFrom );
//	J_to.block<6,1>(0,4) = dpose_dT_times_dT_ddelta( poseConstraintTransform.inverse() * poseEstimateTo.inverse() * -dT_qy_ * refFrameTransform * poseEstimateFrom );
//	J_to.block<6,1>(0,5) = dpose_dT_times_dT_ddelta( poseConstraintTransform.inverse() * poseEstimateTo.inverse() * -dT_qz_ * refFrameTransform * poseEstimateFrom );
//
//}
//
//
//bool MultiFrameMultiResolutionColorSurfelRegistration::estimateTransformationLevenbergMarquardt( unsigned int maxIterations ) {
//
//	const bool useFeatures = true;
//
//	const double tau = 10e-5;
//	const double min_delta = 1e-3;
//	const double min_error = 1e-6;
//
//	float minResolution = std::min( start_resolution_, stop_resolution_ );
//	float maxResolution = std::max( start_resolution_, stop_resolution_ );
//
//	pcl::StopWatch stopwatch;
//
//	std::vector< Eigen::Matrix< double, 6, 1 >, Eigen::aligned_allocator< Eigen::Matrix< double, 6, 1 > > > pose_estimate( registration_pairs_.size() );
//	std::vector< double > fs( registration_pairs_.size() );
//	std::vector< Eigen::Matrix< double, 6, 1 >, Eigen::aligned_allocator< Eigen::Matrix< double, 6, 1 > > > dfs( registration_pairs_.size() );
//	std::vector< Eigen::Matrix< double, 6, 6 >, Eigen::aligned_allocator< Eigen::Matrix< double, 6, 6 > > > d2fs( registration_pairs_.size() );
//
//	const Eigen::Matrix< double, 6, 6 > id6 = Eigen::Matrix< double, 6, 6 >::Identity();
//	double mu = -1.0;
//	double nu = 2;
//
//	double last_error = std::numeric_limits<double>::max();
//
//	bool reassociate = true;
//
//	bool reevaluateGradient = true;
//
//
//	bool retVal = true;
//
//	int iter = 0;
//	while( iter < maxIterations ) {
//
//		retVal = true;
//
//		std::vector< double > lastWSign( 6*registration_pairs_.size(), 1 );
//
//		for( unsigned int i = 0; i < registration_pairs_.size(); i++ ) {
//
//			FramePair& pair = registration_pairs_[i];
//
//			// set up the minimization algorithm
//			MultiResolutionColorSurfelRegistration::Params params;
//			params. = minResolution;
//			params.maxResolution = maxResolution;
//			params.transform = &pair.transform_;
//
//			Eigen::Quaterniond q( pair.transform_.block<3,3>(0,0) );
//			params.lastWSign = q.w() / fabsf(q.w());
//			lastWSign[i] = round(params.lastWSign);
//
//			// initialize pose_estimate
//			pose_estimate[i](0) = pair.transform_(0,3);
//			pose_estimate[i](1) = pair.transform_(1,3);
//			pose_estimate[i](2) = pair.transform_(2,3);
//			pose_estimate[i](3) = q.x();
//			pose_estimate[i](4) = q.y();
//			pose_estimate[i](5) = q.z();
//
//
//			MultiResolutionColorSurfelRegistration::SurfelAssociationList surfelAssociations;
//
//			if( reevaluateGradient ) {
//
//				if( reassociate ) {
//					pair.target_->clearAssociations();
//				}
//
//				float searchDistFactor = 2.f;
//				float maxSearchDist = 2.f*maxResolution;
//
//				stopwatch.reset();
//				surfelAssociations.clear();
//				associateMapsBreadthFirstParallel( surfelAssociations, *pair.source_, *pair.target_, target_sampling_maps_[i], pair.transform_, 0.99f*minResolution, 1.01f*maxResolution, searchDistFactor, maxSearchDist, useFeatures );
//				double deltat = stopwatch.getTime();
//
//				stopwatch.reset();
//				retVal = retVal && registrationSurfelErrorFunctionWithFirstAndSecondDerivativeLM( pose_estimate[i], params, fs[i], dfs[i], d2fs[i], surfelAssociations );
//
//
//				double deltat2 = stopwatch.getTime();
//
//			}
//
//		}
//
//
//		reevaluateGradient = false;
//
//		if( !retVal ) {
//			std::cout << "registration failed\n";
//			return false;
//		}
//
//
//		// incorporate pose constraints between targets
//		double pose_constraints_error = 0.0;
//		for( unsigned int i = 0; i < target_pose_constraints_.size(); i++ ) {
//
//			Eigen::Matrix< double, 6, 1 > pose_diff = Eigen::Matrix< double, 6, 1 >::Zero();
//			Eigen::Matrix< double, 6, 6 > J_from = Eigen::Matrix< double, 6, 6 >::Zero();
//			Eigen::Matrix< double, 6, 6 > J_to = Eigen::Matrix< double, 6, 6 >::Zero();
//
//			poseConstraintErrorWithFirstDerivative( pose_diff, J_from, J_to, target_pose_constraints_[i].ref_frame_transform_, target_pose_constraints_[i].pose_constraint_mean_, registration_pairs_[target_pose_constraint_ids_[i].first].transform_, registration_pairs_[target_pose_constraint_ids_[i].second].transform_ );
//			pose_constraints_error += (pose_diff.transpose() * target_pose_constraints_[i].pose_constraint_invcov_ * pose_diff)(0);
//
//
//			dfs[target_pose_constraint_ids_[i].first] += J_from.transpose() * target_pose_constraints_[i].pose_constraint_invcov_ * -pose_diff;
//			d2fs[target_pose_constraint_ids_[i].first] += J_from.transpose() * target_pose_constraints_[i].pose_constraint_invcov_ * J_from;
//
//			dfs[target_pose_constraint_ids_[i].second] += J_to.transpose() * target_pose_constraints_[i].pose_constraint_invcov_ * -pose_diff;
//			d2fs[target_pose_constraint_ids_[i].second] += J_to.transpose() * target_pose_constraints_[i].pose_constraint_invcov_ * J_to;
//		}
//
//
//		// determine total error and derivatives
//		Eigen::MatrixXd x( 6*registration_pairs_.size(), 1 );
//		Eigen::MatrixXd x_new( 6*registration_pairs_.size(), 1 );
//		Eigen::MatrixXd df( 6*registration_pairs_.size(), 1 );
//		Eigen::MatrixXd d2f( 6*registration_pairs_.size(), 6*registration_pairs_.size() );
//		df.setZero();
//		d2f.setZero();
//		last_error = pose_constraints_error;
//		for( unsigned int i = 0; i < registration_pairs_.size(); i++ ) {
//			last_error += fs[i];
//			x.block<6,1>(6*i,0) = pose_estimate[i];
//			df.block<6,1>(6*i,0) = dfs[i];
//			d2f.block<6,6>(6*i,6*i) = d2fs[i];
//		}
//
//		x_new = x;
//
//
//
////		if( last_error < min_error ) {
//////			std::cout << "converged\n";
////			break;
////		}
//
//
//		if( mu < 0 ) {
//			mu = tau * std::max( d2f.maxCoeff(), -d2f.minCoeff() );
//		}
//
////		std::cout << iter << " mu: " << mu << "\n";
//
//
//		Eigen::MatrixXd id6n = Eigen::MatrixXd::Identity( 6*registration_pairs_.size(), 6*registration_pairs_.size() );
//		Eigen::MatrixXd delta_x = Eigen::MatrixXd::Zero( 6*registration_pairs_.size(), 1 );
//		Eigen::MatrixXd d2f_inv( 6*registration_pairs_.size(), 6*registration_pairs_.size() );
//		if( fabsf( d2f.determinant() ) > std::numeric_limits<double>::epsilon() ) {
//
//			d2f_inv = (d2f + mu * id6n).inverse();
//
//			delta_x = d2f_inv * df;
//
//		}
//
//		if( delta_x.norm() < min_delta ) {
//
//			if( reassociate )
//				break;
//
//			reassociate = true;
//			reevaluateGradient = true;
////			std::cout << "reassociating!\n";
//		}
//		else
//			reassociate = false;
//
//
//		// apply delta_x to each pose estimate
//		for( unsigned int i = 0; i < registration_pairs_.size(); i++ ) {
//
//			double qx = x( i*6 + 3 );
//			double qy = x( i*6 + 4 );
//			double qz = x( i*6 + 5 );
//			double qw = lastWSign[i]*sqrt(1.0-qx*qx-qy*qy-qz*qz);
//
//			Eigen::Matrix4d currentTransform = Eigen::Matrix4d::Identity();
//			currentTransform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
//			currentTransform(0,3) = x( i*6 + 0 );
//			currentTransform(1,3) = x( i*6 + 1 );
//			currentTransform(2,3) = x( i*6 + 2 );
//
//			qx = delta_x( i*6 + 3 );
//			qy = delta_x( i*6 + 4 );
//			qz = delta_x( i*6 + 5 );
//			qw = sqrt(1.0-qx*qx-qy*qy-qz*qz);
//
//			Eigen::Matrix4d deltaTransform = Eigen::Matrix4d::Identity();
//			deltaTransform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
//			deltaTransform(0,3) = delta_x( i*6 + 0 );
//			deltaTransform(1,3) = delta_x( i*6 + 1 );
//			deltaTransform(2,3) = delta_x( i*6 + 2 );
//
//			Eigen::Matrix4d newTransform = deltaTransform * currentTransform;
//
////			Eigen::Matrix< double, 6, 1 > x_new;
//			x_new( i*6 + 0 ) = newTransform(0,3);
//			x_new( i*6 + 1 ) = newTransform(1,3);
//			x_new( i*6 + 2 ) = newTransform(2,3);
//
//			Eigen::Quaterniond q_new( newTransform.block<3,3>(0,0) );
//			x_new( i*6 + 3 ) = q_new.x();
//			x_new( i*6 + 4 ) = q_new.y();
//			x_new( i*6 + 5 ) = q_new.z();
//
//			// write back estimated transforms to frame pairs for reevaluation
//			registration_pairs_[i].transform_ = newTransform;
//		}
//
//
//
//
////		std::cout << "iter: " << iter << ": " << delta_x.norm() << "\n";
//
////		std::cout << x_new.transpose() << "\n";
////
//		double new_error = 0.0;
//		bool retVal2 = true;
//
//		for( unsigned int i = 0; i < registration_pairs_.size(); i++ ) {
//
//			FramePair& pair = registration_pairs_[i];
//
//			// set up the minimization algorithm
//			MultiResolutionColorSurfelRegistration::RegistrationFunctionParameters params;
//			params.source = pair.source_;
//			params.target = pair.target_;
//			params.minResolution = minResolution;
//			params.maxResolution = maxResolution;
//			params.transform = &pair.transform_;
//
//			Eigen::Quaterniond q( pair.transform_.block<3,3>(0,0) );
//			params.lastWSign = q.w() / fabsf(q.w());
//			lastWSign[i] = round(params.lastWSign);
//
//			// initialize pose_estimate
//			pose_estimate[i](0) = pair.transform_(0,3);
//			pose_estimate[i](1) = pair.transform_(1,3);
//			pose_estimate[i](2) = pair.transform_(2,3);
//			pose_estimate[i](3) = q.x();
//			pose_estimate[i](4) = q.y();
//			pose_estimate[i](5) = q.z();
//
//			MultiResolutionColorSurfelRegistration::SurfelAssociationList surfelAssociations;
//
//			if( reevaluateGradient ) {
//
//				if( reassociate ) {
//					pair.target_->clearAssociations();
//				}
//
//				float searchDistFactor = 2.f;
//				float maxSearchDist = 2.f*maxResolution;
//
//				stopwatch.reset();
//				surfelAssociations.clear();
//				associateMapsBreadthFirstParallel( surfelAssociations, *pair.source_, *pair.target_, target_sampling_maps_[i], pair.transform_, 0.99f*minResolution, 1.01f*maxResolution, searchDistFactor, maxSearchDist, useFeatures );
//				double deltat = stopwatch.getTime();
//
//				stopwatch.reset();
//				double f_new = 0.0;
//				retVal2 = retVal2 && registrationErrorFunctionLM( pose_estimate[i], params, f_new, surfelAssociations );
//				new_error += f_new;
//				double deltat2 = stopwatch.getTime();
//
//			}
//
//		}
//
//		if( !retVal2 )
//			return false;
//
//		// incorporate pose constraints between targets in reevaluated error function
//		for( unsigned int i = 0; i < target_pose_constraints_.size(); i++ ) {
//
//			Eigen::Matrix< double, 6, 1 > pose_diff = Eigen::Matrix< double, 6, 1 >::Zero();
//
//			poseConstraintError( pose_diff, target_pose_constraints_[i].ref_frame_transform_, target_pose_constraints_[i].pose_constraint_mean_, registration_pairs_[target_pose_constraint_ids_[i].first].transform_, registration_pairs_[target_pose_constraint_ids_[i].second].transform_ );
//			new_error += (pose_diff.transpose() * target_pose_constraints_[i].pose_constraint_invcov_ * pose_diff)(0);
//
//		}
//
//
//		Eigen::MatrixXd tmp = delta_x.transpose() * (mu * delta_x + df);
////		std::cout << delta_x.transpose() << "\n";
////		std::cout << (mu * delta_x + df).transpose() << "\n";
////		std::cout << tmp(0,0) << " " << last_error << " " << new_error << "\n";
//
//		double rho = (last_error - new_error) / tmp(0,0);
//
////		std::cout << "rho: " << rho << ", nu: " << nu << "\n";
//
//		if( rho > 0 ) {
//
//			x = x_new;
//
//			mu *= std::max( 0.333, 1.0 - pow( 2.0*rho-1.0, 3.0 ) );
//			nu = 2;
//
//			reevaluateGradient = true;
//
//		}
//		else {
//
//			mu *= nu; nu *= 2.0;
//
//		}
//
//
//		for( unsigned int i = 0; i < registration_pairs_.size(); i++ ) {
//
//			double qx = x( i*6 + 3 );
//			double qy = x( i*6 + 4 );
//			double qz = x( i*6 + 5 );
//			double qw = lastWSign[i]*sqrt(1.0-qx*qx-qy*qy-qz*qz);
//
//			if( isnan(qw) || fabsf(qx) > 1.f || fabsf(qy) > 1.f || fabsf(qz) > 1.f ) {
//				return false;
//			}
//
//			// write back estimated transforms to frame pairs
//			Eigen::Matrix4d currentTransform = Eigen::Matrix4d::Identity();
//			currentTransform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
//			currentTransform(0,3) = x( i*6 + 0 );
//			currentTransform(1,3) = x( i*6 + 1 );
//			currentTransform(2,3) = x( i*6 + 2 );
//
//			registration_pairs_[i].transform_ = currentTransform;
//
//		}
//
//
//
//
//		last_error = new_error;
//
//		iter++;
//
//	}
//
//
//
//
//	return retVal;
//
//}
//
//bool MultiFrameMultiResolutionColorSurfelRegistration::estimateTransformationNewton( unsigned int maxIterations ) {
//
//	return true;
//
//}
//
//bool MultiFrameMultiResolutionColorSurfelRegistration::estimateTransformation( unsigned int levmarIterations, unsigned int newtonIterations ) {
//
//	target_sampling_maps_.clear();
//	target_sampling_maps_.resize( registration_pairs_.size() );
//	for( unsigned int i = 0; i < registration_pairs_.size(); i++ ) {
//		target_sampling_maps_[i] = algorithm::downsampleVectorOcTree(*registration_pairs_[i].target_->octree_, false, registration_pairs_[i].target_->octree_->max_depth_);
//		registration_pairs_[i].target_->clearAssociations();
//	}
//
//	bool retVal = true;
//	if( levmarIterations > 0 )
//		retVal = estimateTransformationLevenbergMarquardt( levmarIterations );
//
//	if( !retVal )
//		std::cout << "levenberg marquardt failed\n";
//
//	std::vector< Eigen::Matrix4d, Eigen::aligned_allocator< Eigen::Matrix4d > > levmarTransforms;
//	for( unsigned int i = 0; i < registration_pairs_.size(); i++ )
//		levmarTransforms.push_back( registration_pairs_[i].transform_ );
//
//	if( retVal ) {
//
//		bool retVal2 = estimateTransformationNewton( newtonIterations );
//		if( !retVal2 ) {
//			std::cout << "newton failed\n";
//
//			for( unsigned int i = 0; i < registration_pairs_.size(); i++ )
//				registration_pairs_[i].transform_ = levmarTransforms[i];
//
//			if( levmarIterations == 0 )
//				retVal = false;
//		}
//
//	}
//
//	return retVal;
//
//}
//
//
