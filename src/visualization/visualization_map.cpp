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

#include "mrsmap/visualization/visualization_map.h"

#include "pcl/common/common_headers.h"

#include "pcl/common/transforms.h"

#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl/surface/grid_projection.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/ear_clipping.h>
#include <pcl/surface/poisson.h>

#include <pcl/search/flann_search.h>

#include <boost/make_shared.hpp>

#include <vtkSurfaceReconstructionFilter.h>
#include <vtkMarchingCubes.h>
#include <vtkReverseSense.h>
#include <vtkContourFilter.h>
#include <vtkGaussianSplatter.h>
#include <vtkCleanPolyData.h>
#include <vtkDelaunay2D.h>
#include <vtkDelaunay3D.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>

using namespace mrsmap;

Viewer::Viewer() {

	viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>( new pcl::visualization::PCLVisualizer( "MRS Viewer" ) );
	viewer->setBackgroundColor( 1, 1, 1 );
//	viewer->setBackgroundColor( 0,0,0 );
//	viewer->addCoordinateSystem( 0.2 );
	viewer->initCameraParameters();

	viewer->registerKeyboardCallback( &Viewer::keyboardEventOccurred, *this, NULL );
	viewer->registerPointPickingCallback( &Viewer::pointPickingCallback, *this, NULL );

	selectedDepth = -1; // d
	selectedViewDir = -1; // v
	processFrame = true; // p
	displayScene = true; // s
	displayMap = true; // m
	displayCorr = false; // c
	displayAll = true; // a
	displayFeatureSimilarity = false; // F
	recordFrame = false; // r
	forceRedraw = false; // f

	is_running = true;

}

Viewer::~Viewer() {
}


void Viewer::spinOnce() {

	if( !viewer->wasStopped() ) {
		viewer->spinOnce(1);
	}
	else {
		is_running = false;
	}

}


void Viewer::pointPickingCallback( const pcl::visualization::PointPickingEvent& event, void* data ) {

	pcl::PointXYZ p;
	event.getPoint( p.x, p.y, p.z );
	std::cout << "picked " << p.x << " " << p.y << " " << p.z << "\n";

	selectedPoint = p;

	viewer->removeShape( "selected_point" );
	viewer->addSphere( p, 0.025, 0, 1.0, 0, "selected_point" );

}

void Viewer::displayPointCloud( const std::string& name, const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, int pointSize ) {

	pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud2 = pcl::PointCloud< pcl::PointXYZRGB >::Ptr( new pcl::PointCloud< pcl::PointXYZRGB >() );
	pcl::copyPointCloud( *cloud, *cloud2 );

	for( unsigned int i = 0; i < cloud2->points.size(); i++ )
		if( isnan( cloud2->points[i].x ) ) {
			cloud2->points[i].x = 0;
			cloud2->points[i].y = 0;
			cloud2->points[i].z = 0;
		}

	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb = pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>( cloud2 );

	if( !viewer->updatePointCloud<pcl::PointXYZRGB>( cloud2, rgb, name ) ) {
		viewer->addPointCloud<pcl::PointXYZRGB>( cloud2, rgb, name );
	}
	viewer->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointSize, name );
}




void Viewer::displayPose( const Eigen::Matrix4d& pose ) {

	static int poseidx = 0;

	double axislength = 0.2;

	pcl::PointXYZRGB p1, p2;

	char str[255];

	if( poseidx > 0 ) {
		sprintf( str, "posex%i", poseidx-1 );
		viewer->removeShape( str );
	}
	sprintf( str, "posex%i", poseidx );
	p1.x = pose(0,3);
	p1.y = pose(1,3);
	p1.z = pose(2,3);
	p2.x = p1.x + axislength*pose(0,0);
	p2.y = p1.y + axislength*pose(1,0);
	p2.z = p1.z + axislength*pose(2,0);
	viewer->addLine( p1, p2, 1.0, 0.0, 0.0, str );
	viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, str );


	if( poseidx > 0 ) {
		sprintf( str, "posey%i", poseidx-1 );
		viewer->removeShape( str );
	}
	sprintf( str, "posey%i", poseidx );
	p1.x = pose(0,3);
	p1.y = pose(1,3);
	p1.z = pose(2,3);
	p2.x = p1.x + axislength*pose(0,1);
	p2.y = p1.y + axislength*pose(1,1);
	p2.z = p1.z + axislength*pose(2,1);
	viewer->addLine( p1, p2, 0.0, 1.0, 0.0, str );
	viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, str );



	if( poseidx > 0 ) {
		sprintf( str, "posez%i", poseidx-1 );
		viewer->removeShape( str );
	}
	sprintf( str, "posez%i", poseidx );
	p1.x = pose(0,3);
	p1.y = pose(1,3);
	p1.z = pose(2,3);
	p2.x = p1.x + axislength*pose(0,2);
	p2.y = p1.y + axislength*pose(1,2);
	p2.z = p1.z + axislength*pose(2,2);
	viewer->addLine( p1, p2, 0.0, 0.0, 1.0, str );
	viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, str );

	poseidx++;

}


void Viewer::displayPose( const std::string& name, const Eigen::Matrix4d& pose ) {

	double axislength = 0.2;

	pcl::PointXYZRGB p1, p2;

	char str[255];

	sprintf( str, "%sposex", name.c_str() );
	viewer->removeShape( str );
	sprintf( str, "%sposex", name.c_str() );
	p1.x = pose(0,3);
	p1.y = pose(1,3);
	p1.z = pose(2,3);
	p2.x = p1.x + axislength*pose(0,0);
	p2.y = p1.y + axislength*pose(1,0);
	p2.z = p1.z + axislength*pose(2,0);
	viewer->addLine( p1, p2, 1.0, 0.0, 0.0, str );
	viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, str );


	sprintf( str, "%sposey", name.c_str() );
	viewer->removeShape( str );
	sprintf( str, "%sposey", name.c_str() );
	p1.x = pose(0,3);
	p1.y = pose(1,3);
	p1.z = pose(2,3);
	p2.x = p1.x + axislength*pose(0,1);
	p2.y = p1.y + axislength*pose(1,1);
	p2.z = p1.z + axislength*pose(2,1);
	viewer->addLine( p1, p2, 0.0, 1.0, 0.0, str );
	viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, str );



	sprintf( str, "%sposez", name.c_str() );
	viewer->removeShape( str );
	sprintf( str, "%sposez", name.c_str() );
	p1.x = pose(0,3);
	p1.y = pose(1,3);
	p1.z = pose(2,3);
	p2.x = p1.x + axislength*pose(0,2);
	p2.y = p1.y + axislength*pose(1,2);
	p2.z = p1.z + axislength*pose(2,2);
	viewer->addLine( p1, p2, 0.0, 0.0, 1.0, str );
	viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, str );


}


void Viewer::displayCorrespondences( const std::string& name, const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud1, const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud2, const Eigen::Matrix4f& transform ) {

//	char str[255];
//
//	for( unsigned int i = 0; i < currShapes.size(); i++ ) {
//		sprintf( str, "shape%i", currShapes[i] );
//		viewer->removeShape( str );
//	}
//	currShapes.clear();
//
//	Eigen::Affine3f transforma( transform );
//
//	for( unsigned int i = 0; i < cloud1->points.size(); i++ ) {
//
//		currShapes.push_back( shapeIdx );
//		sprintf( str, "shape%i", shapeIdx++ );
//
//		float r = cloud1->points[i].r / 255.f;
//		float g = cloud1->points[i].g / 255.f;
//		float b = cloud1->points[i].b / 255.f;
//
//		viewer->addLine( pcl::transformPoint( cloud1->points[i], transforma ), pcl::transformPoint( cloud2->points[i], transforma ), r, g, b, str );
//		viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, str );
//
//		std::cout << i << "\n";
//
//	}

	Eigen::Affine3f transforma( transform );
	pcl::Correspondences corr;
	for( unsigned int i = 0; i < cloud1->points.size(); i++ ) {

		pcl::Correspondence c( i, i, 0 );
		corr.push_back( c );

	}

	viewer->addCorrespondences<pcl::PointXYZRGB>( cloud1, cloud2, corr, name );

}

void Viewer::removeCorrespondences( const std::string& name ) {

//	char str[255];
//
//	for( unsigned int i = 0; i < currShapes.size(); i++ ) {
//		sprintf( str, "shape%i", currShapes[i] );
//		viewer->removeShape( str );
//	}
//	currShapes.clear();

	viewer->removeCorrespondences( name );

}


void Viewer::displaySurfaceNormals( const std::string& name, const pcl::PointCloud< pcl::PointXYZRGBNormal >::Ptr& cloud ) {

	viewer->setBackgroundColor( 0,0,0 );

//	char str[255];
//
//	for( unsigned int i = 0; i < currNormals.size(); i++ ) {
//		sprintf( str, "normal%i", currNormals[i] );
//		viewer->removeShape( str );
//	}
//	currNormals.clear();
//
////	Eigen::Affine3f transforma( transform );
//
//	for( unsigned int i = 0; i < cloud->points.size(); i++ ) {
//
//		currNormals.push_back( normalIdx );
//		sprintf( str, "normal%i", normalIdx++ );
//
//		pcl::PointXYZ p1, p2;
//
//		const pcl::PointNormal& p = cloud->points[i];
//
//		p1.x = p.x;
//		p1.y = p.y;
//		p1.z = p.z;
//
//		p2.x = p.x + 0.1*p.normal_x;
//		p2.y = p.y + 0.1*p.normal_y;
//		p2.z = p.z + 0.1*p.normal_z;
//
//		viewer->addLine( p1, p2, 1, 0, 0, str );
//		viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, str );
//
//	}

	if( !viewer->addPointCloudNormals< pcl::PointXYZRGBNormal >( cloud, 1, 0.1, name+"normals" ) ) {
		viewer->removePointCloud(name+"normals");
		viewer->addPointCloudNormals< pcl::PointXYZRGBNormal >( cloud, 1, 0.1, name+"normals" );
	}

	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb = pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>( cloud );

	if( !viewer->updatePointCloud<pcl::PointXYZRGBNormal>( cloud, rgb, name ) ) {
		viewer->addPointCloud<pcl::PointXYZRGBNormal>( cloud, rgb, name );
	}
	viewer->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, name );

}

void Viewer::removeSurfaceNormals( const std::string& name ) {

	char str[255];

	for( unsigned int i = 0; i < currNormals.size(); i++ ) {
		sprintf( str, "normal%i", currNormals[i] );
		viewer->removeShape( str );
	}
	currNormals.clear();

}

vtkSmartPointer<vtkPolyData> transform_back(vtkSmartPointer<vtkPoints> pt, vtkSmartPointer<vtkPolyData> pd)
{
	// The reconstructed surface is transformed back to where the
	// original points are. (Hopefully) it is only a similarity
	// transformation.

	// 1. Get bounding box of pt, get its minimum corner (left, bottom, least-z), at c0, pt_bounds

	// 2. Get bounding box of surface pd, get its minimum corner (left, bottom, least-z), at c1, pd_bounds

	// 3. compute scale as:
	//       scale = (pt_bounds[1] - pt_bounds[0])/(pd_bounds[1] - pd_bounds[0]);

	// 4. transform the surface by T := T(pt_bounds[0], [2], [4]).S(scale).T(-pd_bounds[0], -[2], -[4])



	// 1.
	double pt_bounds[6];  // (xmin,xmax, ymin,ymax, zmin,zmax)
	pt->GetBounds(pt_bounds);


	// 2.
	double pd_bounds[6];  // (xmin,xmax, ymin,ymax, zmin,zmax)
	pd->GetBounds(pd_bounds);

	//   // test, make sure it is isotropic
	//   std::cout<<(pt_bounds[1] - pt_bounds[0])/(pd_bounds[1] - pd_bounds[0])<<std::endl;
	//   std::cout<<(pt_bounds[3] - pt_bounds[2])/(pd_bounds[3] - pd_bounds[2])<<std::endl;
	//   std::cout<<(pt_bounds[5] - pt_bounds[4])/(pd_bounds[5] - pd_bounds[4])<<std::endl;
	//   // TEST


	// 3
	double scale = (pt_bounds[1] - pt_bounds[0])/(pd_bounds[1] - pd_bounds[0]);


	// 4.
	vtkSmartPointer<vtkTransform> transp = vtkSmartPointer<vtkTransform>::New();
	transp->Translate(pt_bounds[0], pt_bounds[2], pt_bounds[4]);
	transp->Scale(scale, scale, scale);
	transp->Translate(- pd_bounds[0], - pd_bounds[2], - pd_bounds[4]);

	vtkSmartPointer<vtkTransformPolyDataFilter> tpd = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
	#if VTK_MAJOR_VERSION <= 5
	tpd->SetInput(pd);
	#else
	tpd->SetInputData(pd);
	#endif
	tpd->SetTransform(transp);
	tpd->Update();


	return tpd->GetOutput();
}


void Viewer::displayMesh( const std::string& name, const pcl::PointCloud< pcl::PointXYZRGBNormal >::Ptr& cloud, float resolution, bool poisson ) {

//	pcl::PointCloud< pcl::PointXYZRGBNormal >::Ptr meshCloud = pcl::PointCloud< pcl::PointXYZRGBNormal >::Ptr( new pcl::PointCloud< pcl::PointXYZRGBNormal >() );
//	boost::shared_ptr< std::vector< pcl::Vertices > > vertices = boost::make_shared< std::vector< pcl::Vertices > >();
//	pcl::KdTreeFLANN< pcl::PointXYZRGBNormal > kdtree;
//	kdtree.setInputCloud( cloud );
//
//////	vtkSmartPointer<vtkActor> surfaceActor = vtkSmartPointer<vtkActor>::New();
//////	if( mesh_actor_map_.find( name ) != mesh_actor_map_.end() ) {
//////		surfaceActor = mesh_actor_map_.find( name )->second;
//////	}
////
////	pcl::visualization::PointCloudGeometryHandlerXYZ<pcl::PointXYZRGBNormal>::Ptr geometry_handler( new pcl::visualization::PointCloudGeometryHandlerXYZ<pcl::PointXYZRGBNormal>( cloud ) );
////
////	vtkSmartPointer< vtkPoints > points;
////	geometry_handler->getGeometry( points );
////
////	vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
////	polydata->SetPoints(points);
////
////	vtkSmartPointer<vtkPolyData> outputPolydata = vtkSmartPointer<vtkPolyData>::New();
////
//////	  vtkSmartPointer<vtkGaussianSplatter> splatter = vtkSmartPointer<vtkGaussianSplatter>::New();
//////	  splatter->SetInput(polydata);
//////	  splatter->SetSampleDimensions(50,50,50);
//////	  splatter->SetRadius(0.1);
//////	  splatter->ScalarWarpingOff();
//////
//////
//////	  vtkSmartPointer<vtkContourFilter> contourFilter = vtkSmartPointer<vtkContourFilter>::New();
//////	  contourFilter->SetInputConnection(splatter->GetOutputPort());
//////	  contourFilter->SetValue(0, 0.01);
////
//////	  // Sometimes the contouring algorithm can create a volume whose gradient
//////	  // vector and ordering of polygon (using the right hand rule) are
//////	  // inconsistent. vtkReverseSense cures this problem.
//////	  vtkSmartPointer<vtkReverseSense> reverse = vtkSmartPointer<vtkReverseSense>::New();
//////	  reverse->SetInputConnection(contourFilter->GetOutputPort());
//////	  reverse->ReverseCellsOn();
//////	  reverse->ReverseNormalsOn();
//////	  reverse->Update();
////
////	vtkSmartPointer< vtkSurfaceReconstructionFilter > srf = vtkSmartPointer< vtkSurfaceReconstructionFilter >::New();
////	srf->SetSampleSpacing( 0.5*resolution );
////	srf->SetInput( polydata );
////
////
////	vtkSmartPointer< vtkMarchingCubes > mc = vtkSmartPointer< vtkMarchingCubes >::New();
//////	mc->SetComputeScalars(1);
//////	mc->SetComputeNormals(1);
//////	mc->SetComputeGradients(1);
////	mc->SetValue( 0, 0.0 );
//////	mc->GenerateValues(1, -0.1, 0.1);
////	mc->SetOutput(outputPolydata);
////	mc->SetInputConnection( srf->GetOutputPort() );
////	mc->Update();
////
////	vtkSmartPointer<vtkPolyDataMapper> map = vtkSmartPointer<vtkPolyDataMapper>::New();
////	map->SetInputConnection(mc->GetOutputPort());
//////	map->ScalarVisibilityOff();
////
////
////
////
//////	  // Clean the polydata. This will remove duplicate points that may be
//////	  // present in the input data.
//////	  vtkSmartPointer<vtkCleanPolyData> cleaner =
//////	    vtkSmartPointer<vtkCleanPolyData>::New();
//////	  cleaner->SetInput (polydata);
//////
//////	  // Generate a tetrahedral mesh from the input points. By
//////	  // default, the generated volume is the convex hull of the points.
//////	  vtkSmartPointer<vtkDelaunay2D> delaunay2D =
//////	    vtkSmartPointer<vtkDelaunay2D>::New();
////////	  delaunay3D->SetInputConnection (cleaner->GetOutputPort());
//////	  delaunay2D->SetAlpha( resolution );
//////	  delaunay2D->SetInput( polydata );
//////	  delaunay2D->Update();
////
//////	  vtkSmartPointer<vtkDataSetMapper> delaunayMapper =
//////	    vtkSmartPointer<vtkDataSetMapper>::New();
//////	  delaunayMapper->SetInputConnection(delaunay2D->GetOutputPort());
////
//////	surfaceActor->SetMapper(map);
//////	surfaceActor->GetProperty()->SetDiffuseColor(1.0000, 0.3882, 0.2784);
//////	surfaceActor->GetProperty()->SetSpecularColor(1, 1, 1);
//////	surfaceActor->GetProperty()->SetSpecular(.4);
//////	surfaceActor->GetProperty()->SetSpecularPower(50);
//////
//////	if( mesh_actor_map_.find( name ) == mesh_actor_map_.end() ) {
//////		viewer->addActorToRenderer( surfaceActor );
//////		mesh_actor_map_[name] = surfaceActor;
//////	}
////
////	vtkSmartPointer<vtkPolyData> surfPoly = transform_back( points, mc->GetOutput());
////	vtkCellArray* surfPolys = surfPoly->GetPolys();
////	vtkPoints* surfPoints = surfPoly->GetPoints();
////
////	std::vector< int > usedPoints( surfPoints->GetNumberOfPoints(), -1 );
////	for( unsigned int i = 0; i < usedPoints.size(); i++ ) {
////
////		double* pptr = surfPoints->GetPoint( i );
////
////		pcl::PointXYZRGBNormal p;
////		p.x = pptr[0];
////		p.y = pptr[1];
////		p.z = pptr[2];
////
////		std::vector< int > k_indices( 1, -1 );
////		std::vector< float > k_sqr_distances( 1, 0.f );
////		int found = kdtree.nearestKSearch( p, 1, k_indices, k_sqr_distances );
////		if( found ) {
////			if( k_sqr_distances[0] < maxDist2 ) {
////				// fill in mesh cloud (with correct RGB)
////				p.rgb = cloud->points[k_indices[0]].rgb;
////				meshCloud->points.push_back( p );
////				usedPoints[i] = meshCloud->points.size()-1;
////			}
////		}
////
////	}
////
////	vtkIdType npts;
////	vtkIdType* ptsPtr;
////	surfPolys->InitTraversal();
////	while( 	surfPolys->GetNextCell( npts, ptsPtr ) ) {
////
////		// check if all point indices are in the map of used points and fill in vertices list
////		bool polyOK = true;
////		for( unsigned int i = 0; i < npts; i++ ) {
////			if( usedPoints[ptsPtr[i]] < 0 ) {
////				polyOK = false;
////			}
////		}
////
////		if( polyOK ) {
////			pcl::Vertices v;
////			for( unsigned int i = 0; i < npts; i++ ) {
////				v.vertices.push_back( usedPoints[ ptsPtr[i] ] );
////			}
////			vertices->push_back( v );
////		}
////
////	}
//
//
////	if( cloud->points.size() == 0 ) {
////		viewer->removePolygonMesh( name );
////		return;
////	}
////
////	// perform mesh reconstruction on cloud
////
////////	pcl::MarchingCubes< pcl::PointXYZRGBNormal > mc;
////////	pcl::MarchingCubesRBF< pcl::PointXYZRGBNormal > mc;
//////	pcl::MarchingCubesHoppe< pcl::PointXYZRGBNormal > mc;
//////
//////	mc.setIsoLevel( 0.0 );
//////	mc.setGridResolution( 50, 50, 50 );//resolution, resolution, resolution );
//////	mc.setPercentageExtendGrid( 0.1 );
////////	mc.setOffSurfaceDisplacement( 0.01 );
//////	mc.setInputCloud( cloud );
//////
//////	mc.reconstruct( *meshCloud, *vertices );
////
//////	pcl::GridProjection< pcl::PointXYZRGBNormal > gp;
//////	gp.setInputCloud( cloud );
//////	gp.setResolution( resolution );
//////
//////	gp.reconstruct( *meshCloud, *vertices );
////
////
////	pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree = pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr (new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
////	tree->setInputCloud( cloud );
////
////	pcl::GreedyProjectionTriangulation< pcl::PointXYZRGBNormal > gp;
////	gp.setSearchMethod( tree );
////	gp.setMaximumNearestNeighbors( 100 );
////	gp.setSearchRadius( 8.f*resolution );
////	gp.setMu( 2.5 );
//////	gp.setNormalConsistency(true);
//////	gp.setConsistentVertexOrdering(true);
////
//////	gp.setMaximumSurfaceAngle(M_PI);
//////    gp.setMinimumAngle(M_PI/4);
//////    gp.setMaximumAngle(M_PI/2.0);
////
////	gp.setInputCloud( cloud );
////	gp.reconstruct( *vertices );
////	*meshCloud = *cloud;
//
////	// perform "texture mapping"
////	for( unsigned int i = 0; i < meshCloud->points.size(); i++ ) {
////		std::vector< int > k_indices( 1, -1 );
////		std::vector< float > k_sqr_distances( 1, 0.f );
////		int found = kdtree.nearestKSearch( meshCloud->points[i], 1, k_indices, k_sqr_distances );
////		if( found ) {
////			meshCloud->points[i].rgb = cloud->points[k_indices[0]].rgb;
////		}
////	}
//
//
//
//	pcl::PointCloud< pcl::PointXYZRGBNormal >::Ptr meshCloud2 = pcl::PointCloud< pcl::PointXYZRGBNormal >::Ptr( new pcl::PointCloud< pcl::PointXYZRGBNormal >() );
//	boost::shared_ptr< std::vector< pcl::Vertices > > vertices2 = boost::make_shared< std::vector< pcl::Vertices > >();
//	pcl::Poisson< pcl::PointXYZRGBNormal > p;
//	p.setDepth(12);
//	p.setSamplesPerNode( 1 );
//	p.setInputCloud( cloud );
//	p.reconstruct( *meshCloud2, *vertices2 );
//
//	std::vector< int > usedPoints( meshCloud2->points.size(), -1 );
//	for( unsigned int i = 0; i < usedPoints.size(); i++ ) {
//
//		pcl::PointXYZRGBNormal p = meshCloud2->points[i];
//
//		std::vector< int > k_indices( 1, -1 );
//		std::vector< float > k_sqr_distances( 1, 0.f );
//		int found = kdtree.nearestKSearch( p, 1, k_indices, k_sqr_distances );
//		if( found ) {
//
//			std::vector< int > k2_indices( 10, -1 );
//			std::vector< float > k2_sqr_distances( 10, 0.f );
//			int found2 = kdtree.nearestKSearch( cloud->points[k_indices[0]], 10, k2_indices, k2_sqr_distances );
//
//			float maxDist2 = 0;
//			for( unsigned int j = 0; j < found2; j++ ) {
//				maxDist2 += k2_sqr_distances[j];
//			}
//			maxDist2 *= 0.2f / (float)found;
//
//			if( k_sqr_distances[0] < maxDist2 ) {
//				// fill in mesh cloud (with correct RGB)
//				p.rgb = cloud->points[k_indices[0]].rgb;
//				meshCloud->points.push_back( p );
//				usedPoints[i] = meshCloud->points.size()-1;
//			}
//		}
//
//	}
//
//	for( unsigned int j = 0; j < vertices2->size(); j++ ) {
//
//		const pcl::Vertices& v2 = (*vertices2)[j];
//
//		// check if all point indices are in the map of used points and fill in vertices list
//		bool polyOK = true;
//		for( unsigned int i = 0; i < v2.vertices.size(); i++ ) {
//			if( usedPoints[v2.vertices[i]] < 0 ) {
//				polyOK = false;
//			}
//		}
//
//		if( polyOK ) {
//			pcl::Vertices v;
//			for( unsigned int i = 0; i < v2.vertices.size(); i++ ) {
//				v.vertices.push_back( usedPoints[v2.vertices[i]] );
//			}
//			vertices->push_back( v );
//		}
//
//	}
//
//
//	// display mesh
//	if( !viewer->addPolygonMesh< pcl::PointXYZRGBNormal >( meshCloud, *vertices, name ) ) {
//		viewer->removePolygonMesh( name );
//		viewer->addPolygonMesh< pcl::PointXYZRGBNormal >( meshCloud, *vertices, name );
//	}
//
//	(*viewer->cloud_actor_map_)[name].actor->GetProperty()->SetInterpolationToPhong();
//	(*viewer->cloud_actor_map_)[name].actor->GetProperty()->ShadingOn();
////	(*viewer->cloud_actor_map_)[name].actor->GetProperty()->SetDiffuseColor(1.0000, 0.3882, 0.2784);
////	(*viewer->cloud_actor_map_)[name].actor->GetProperty()->SetSpecularColor(1, 1, 1);
////	(*viewer->cloud_actor_map_)[name].actor->GetProperty()->SetSpecular(.4);
////	(*viewer->cloud_actor_map_)[name].actor->GetProperty()->SetSpecularPower(50);
//
////	if( !viewer->updatePolygonMesh< pcl::PointXYZRGBNormal >( cloud, *vertices, name ) ) {
////		viewer->addPolygonMesh< pcl::PointXYZRGBNormal >( cloud, *vertices, name );
////	}

}

void Viewer::keyboardEventOccurred( const pcl::visualization::KeyboardEvent &event, void* data ) {

	if( (event.getKeySym() == "d" || event.getKeySym() == "D") && event.keyDown() ) {

		if( event.getKeySym() == "d" ) {
			selectedDepth++;
		}
		else {
			selectedDepth--;
			if( selectedDepth < 0 )
				selectedDepth = 15;
		}

		selectedDepth = selectedDepth % 16;
		std::cout << "Selected Depth " << selectedDepth << "\n";
	}
	if( (event.getKeySym() == "v" || event.getKeySym() == "V") && event.keyDown() ) {

		if( event.getKeySym() == "v" ) {
			selectedViewDir++;
			if( selectedViewDir == 7 )
				selectedViewDir = -1;
		}
		else {
			selectedViewDir--;
			if( selectedViewDir < -1 )
				selectedViewDir = 6;
		}

		std::cout << "Selected View Dir " << selectedViewDir << "\n";

	}
	if( (event.getKeySym() == "p") && event.keyDown() ) {
		processFrame = true;
	}
	if( (event.getKeySym() == "h") && event.keyDown() ) {
		processFrame = !processFrame;
	}
	if( (event.getKeySym() == "s") && event.keyDown() ) {
		displayScene = !displayScene;
	}
	if( (event.getKeySym() == "m") && event.keyDown() ) {
		displayMap = !displayMap;
	}
	if( (event.getKeySym() == "a") && event.keyDown() ) {
		displayAll = !displayAll;
	}
	if( (event.getKeySym() == "c") && event.keyDown() ) {
		displayCorr = !displayCorr;
	}
	if( (event.getKeySym() == "S") && event.keyDown() ) {
		displayFeatureSimilarity = !displayFeatureSimilarity;
		std::cout << "feature similarity " << (displayFeatureSimilarity ? "on" : "off") << "\n";
	}
	if( (event.getKeySym() == "f" || event.getKeySym() == "r" || event.getKeySym() == "N") && event.keyDown() ) {
		forceRedraw = true;
	}
}





