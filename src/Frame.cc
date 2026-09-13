/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/
#include <pcl/filters/crop_box.h>
#include "Frame.h"
#include "Converter.h"

#include <thread>
//ovd add
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include "Tracking.h"
#include <thread>
#include <chrono>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;


//ovd add
std::vector<cv::Point2f> Prepoint, Curpoint;
cv::Mat imGrayPre,mask;
std::vector<uchar> state;
std::vector<float> Err;
 //为极线距离,epiErr为分布
std::vector<uchar> status;


std::vector<cv::Point2f> pre_box, cur_box, pre_key_box;
std::vector<float> Err_box;
std::vector<uchar> state_box;


namespace ORB_SLAM2 {

    long unsigned int Frame::nNextId = 0;
    bool Frame::mbInitialComputations = true;
    float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
    float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
    float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

    Frame::Frame() {}

//Copy Constructor
    Frame::Frame(const Frame &frame)
            : mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft),
              mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
              mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
              mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
              mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
              mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
              mvDynamicProbability(frame.mvDynamicProbability),
              mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
              mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
              mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
              mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
              mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2),
              mLdesc(frame.mLdesc), NL(frame.NL), mvKeylinesUn(frame.mvKeylinesUn),
              mvpMapLines(frame.mvpMapLines),
              mvbLineOutlier(frame.mvbLineOutlier), mvKeyLineFunctions(frame.mvKeyLineFunctions),
              mvDepthLine(frame.mvDepthLine), mvPlaneCoefficients(frame.mvPlaneCoefficients),
              mbNewPlane(frame.mbNewPlane),
              mvpMapPlanes(frame.mvpMapPlanes), mnPlaneNum(frame.mnPlaneNum), mvbPlaneOutlier(frame.mvbPlaneOutlier),
              mvpParallelPlanes(frame.mvpParallelPlanes), mvpVerticalPlanes(frame.mvpVerticalPlanes),
              mvPlanePoints(frame.mvPlanePoints), mfDisTh(frame.mfDisTh) {
        for (int i = 0; i < FRAME_GRID_COLS; i++)
            for (int j = 0; j < FRAME_GRID_ROWS; j++)
                mGrid[i][j] = frame.mGrid[i][j];

        if (!frame.mTcw.empty())
            SetPose(frame.mTcw);
    }

    Frame::Frame(Tracking* pTracker, const cv::Mat &imRGB, const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp,
                 ORBextractor *extractor, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf,
                 const float &thDepth, const float &depthMapFactor, const float mfDisTh)
            : mpORBvocabulary(voc), mpORBextractorLeft(extractor),
              mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),mpTracker(pTracker),
              mfDisTh(mfDisTh) {
        // Frame ID
        mnId = nNextId++;

        // Scale Level Info
        mnScaleLevels = mpORBextractorLeft->GetLevels();
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        cv::Mat imDepthScaled;
        if (depthMapFactor != 1 || imDepth.type() != CV_32F) {
            imDepth.convertTo(imDepthScaled, CV_32F, depthMapFactor);
        } else {
            imDepthScaled = imDepth;
        }

        //ovd add
        cv::Mat depth_16u;
        if(imDepth.type()!=CV_16UC1)
            imDepth.convertTo(depth_16u, CV_16UC1);
        else
            depth_16u = imDepth.clone();


         mpTracker->semantic_mask = cv::Mat(imDepth.rows, imDepth.cols, CV_16UC1, cv::Scalar(0));
         //mpTracker->semantic_id   = cv::Mat(imDepth.rows, imDepth.cols, CV_16UC1, cv::Scalar(0));

         mpTracker->show_semantic_mask = cv::Mat(imDepth.rows, imDepth.cols, CV_16UC1, cv::Scalar(0));
           mpTracker->compen_semantic = cv::Mat(imDepth.rows, imDepth.cols, CV_16UC1, cv::Scalar(0));
    //     mpTracker->finial_img = cv::Mat(imDepth.rows, imDepth.cols, CV_16UC1, cv::Scalar(0));
    //     auto fx = mpTracker->mK.at<float>(0, 0);       
        
    // // for(int i=0; i<imDepthScaled.rows; i++)
    // // {
    // //     for(int j=0; j<imDepthScaled.cols; j++)
    // //         outFile << imDepthScaled.at<float>(i,j) <<endl;
    // // }
    // // outFile.close();
    // // std::cout << "compen_depth Saved!!" << std::endl;
    //     if(!mpTracker->yoloBoundingBoxList.empty())
    //     {
    //         // record
    //         for(auto it = mpTracker->yoloBoundingBoxList.begin(); it -mpTracker->yoloBoundingBoxList.begin()<mpTracker->yoloBoundingBoxList.size(); it++)
    //         {

    //             if(it->GetLabel()!="person")
    //                 continue;
    //             cv::Rect2f rect_object = it->GetRect();
    //             cv::Point2f p1 = rect_object.tl();
    //             cv::Point2f p2 = rect_object.br();
    //             int x1 = (int) p1.x ;
    //             int y1 = (int) p1.y ;
    //             int x2 = (int) p2.x ;
    //             int y2 = (int) p2.y ;
    //             //std::ofstream outFile("depth_yolo.txt",ios::app);
    //             // if (!outFile) {
    //             //     std::cerr << "无法打cal文件。" << std::endl;
    //             //     return ;
    //             // }
         
    //             mpTracker->setSemanticMask(x1, y1, x2, y2,depth_16u);
    //             for(int i=x1; i<x2; i++)
    //             {
    //                 for(int j=y1; j<y2; j++)
    //                 {
    //                     if(i>=0 && i<imDepthScaled.cols && j>=0 && j<imDepthScaled.rows)
    //                     {
    //                         const float invSigma2 = 128 * imDepthScaled.at<float>(j,i) *
    //                                         imDepthScaled.at<float>(j,i) / (fx);
    //                         float chi = (imDepthScaled.at<float>(j,i) - it->dep_average) / it->dep_stdcov;
    //                         //outFile << imDepthScaled.at<float>(j,i) <<endl;
    //                         //const float invSigma2 = 128 * imDepthScaled.at<float>(j,i) * imDepthScaled.at<float>(j,i) / (fx);
    //                         if( chi * chi < 0.58)
    //                             mpTracker->compen_semantic.at<unsigned short>(j,i) = 0xffff;

    //                     }
    //                 }
    //             } 
               
    //             //outFile.close();
    //             //std::cout << "yolo_depth Saved!!" << std::endl;
    //         }
            
    //         //std::cout << "compen_depth Saved!!" << std::endl;
    //     }
    //     mpTracker->compensateMissedDetection(imDepth.rows, imDepth.cols, depth_16u);
        
    //     for (int i = 0; i < imDepth.rows; i++)
	//     {
	// 	    for (int j = 0; j < imDepth.cols; j++)
	// 	    {
    //             //std::string tempLabel = yoloBoundingBoxList[i].GetLabel();
    //             if(mpTracker->semantic_mask.at<unsigned short>(i,j)>
    //             depth_16u.at<unsigned short>(i,j) && depth_16u.at<unsigned short>(i,j) > 0)
    //                 mpTracker->show_semantic_mask.at<unsigned short>(i,j) = 0xffff;
                
    //             if(mpTracker->show_semantic_mask.at<unsigned short>(i,j)!=0 && mpTracker->compen_semantic.at<unsigned short>(i,j)!=0)
    //                 mpTracker->finial_img.at<unsigned short>(i,j) = 0xffff;
    //         }
    //     }


#ifdef REMOVE_DYNAMIC_OBJECT
    // The detector is fed by System before GrabImage().  Wait for this frame's
    // result before extracting features; the old ordering extracted every ORB
    // feature first and therefore could not implement semantic masking.
    while(!mpTracker->isNewSegmentImgArrived())
    {
        usleep(1);
    }

#if 0 // Superseded by the probability cascade after feature extraction.
    cv::Mat imGrayT = imGray;
    if(imGrayPre.data)
    {
       
        try{
        EpiloarWithFundMatrix(imGrayPre,imGray, imDepthScaled);
        }
        catch(const std::system_error& e)
        {
             std::cerr << "System error at iteration " << ": " << e.what() << std::endl;
        }
        std::swap(imGrayPre,imGrayT);
        
    }
    
    else
        std::swap(imGrayPre,imGrayT);

    // for (int i = 0; i < imDepth.rows; i++)
    // {
    //     for (int j = 0; j < imDepth.cols; j++)
    //     {
    //         //std::string tempLabel = yoloBoundingBoxList[i].GetLabel();
    //         if(mpTracker->semantic_mask.at<unsigned short>(i,j)>
    //         depth_16u.at<unsigned short>(i,j) && depth_16u.at<unsigned short>(i,j) > 0)
    //             mpTracker->show_semantic_mask.at<unsigned short>(i,j) = 0xffff;
    //     }
    // }
    if(!mpTracker->yoloBoundingBoxList.empty())
    {
        // record
        for(auto it = mpTracker->yoloBoundingBoxList.begin(); it -mpTracker->yoloBoundingBoxList.begin()<mpTracker->yoloBoundingBoxList.size(); it++)
        {

            if(it->GetLabel()!="person")
                continue;
            cv::Rect2f rect_object = it->GetRect();
            cv::Point2f p1 = rect_object.tl();
            cv::Point2f p2 = rect_object.br();
            int x1 = (int) p1.x ;
            int y1 = (int) p1.y ;
            int x2 = (int) p2.x ;
            int y2 = (int) p2.y ;
            
            mpTracker->setSemanticMask(x1, y1, x2, y2,depth_16u);
            /*** 
            for(int i=x1; i<x2; i++)
            {
                for(int j=y1; j<y2; j++)
                {
                    if(i>=0 && i<imDepthScaled.cols && j>=0 && j<imDepthScaled.rows)
                    {
                        if(mpTracker->semantic_mask.at<unsigned short>(j,i)>
                        depth_16u.at<unsigned short>(j,i) && depth_16u.at<unsigned short>(j,i) > 0)
                            mpTracker->show_semantic_mask.at<unsigned short>(j,i) = 0xffff;
                
                        // if(mpTracker->show_semantic_mask.at<unsigned short>(i,j)!=0 && mpTracker->compen_semantic.at<unsigned short>(i,j)!=0)
                        //     mpTracker->finial_img.at<unsigned short>(i,j) = 0xffff;

                        float dis = imDepthScaled.at<float>(j,i);
                        if(isnan(dis) || isinf(dis))
                            continue;
                        const float invSigma2 = 128 * imDepthScaled.at<float>(j,i) *
                                        imDepthScaled.at<float>(j,i) / (fx);
                        float chi = (imDepthScaled.at<float>(j,i) - it->key_dep_average) / it->key_dep_stdcov;
                        //cout<<chi *invSigma2*invSigma2* chi<<" ";
                        // if(i%5==0 &&j%5==0)
                        // cout<<chi *invSigma2*invSigma2* chi<<" ";
                        if( chi * chi < 1.9)
                            mpTracker->compen_semantic.at<unsigned short>(j,i) = 0xffff;

                    }
                    //cout<<endl;
                }
            } 
            ***/
        }  
    if(!mvKeys.empty())
    {

        
        //RemoveMovingPointsWithEpiAndYOLO(epiErr,imGray,imDepthScaled);
       
        RemoveMovingKeyPoints(imGray, imDepthScaled);
    
    }
            //outFile.close();
            //std::cout << "yolo_depth Saved!!" << std::endl;
    }
#endif
#endif

        // Extract planes before applying the semantic mask.  They provide the
        // geometric evidence used for the rigid (wall/floor/ceiling) exemption.
        ExtractPlanes(imRGB, imDepth, K, depthMapFactor);

        const cv::Mat staticMask = BuildStaticMask(imGray.size(), imDepthScaled);
        mpTracker->semantic_mask.setTo(cv::Scalar(0));
        mpTracker->semantic_mask.setTo(cv::Scalar(0xffff), staticMask == 0);

        // YOLO detections identify suspects; they no longer hard-mask features.
        thread threadPoints(&ORB_SLAM2::Frame::ExtractORB, this, imGray, cv::Mat());
        thread threadLines(&ORB_SLAM2::Frame::ExtractLSD, this, imGray);
        threadPoints.join();
        threadLines.join();

        // LSD has no mask input. Reject a segment when its midpoint belongs to
        // a dynamic region, keeping line and point observations consistent.
        if(!staticMask.empty() && !mvKeylinesUn.empty()) {
            std::vector<cv::line_descriptor::KeyLine> keptLines;
            std::vector<Eigen::Vector3d> keptLineFunctions;
            cv::Mat keptDescriptors;
            for(size_t i = 0; i < mvKeylinesUn.size(); ++i) {
                const int x = cvRound((mvKeylinesUn[i].startPointX + mvKeylinesUn[i].endPointX) * 0.5f);
                const int y = cvRound((mvKeylinesUn[i].startPointY + mvKeylinesUn[i].endPointY) * 0.5f);
                if(x >= 0 && x < staticMask.cols && y >= 0 && y < staticMask.rows && staticMask.at<uchar>(y,x)) {
                    keptLines.push_back(mvKeylinesUn[i]);
                    if(i < mvKeyLineFunctions.size()) keptLineFunctions.push_back(mvKeyLineFunctions[i]);
                    if(!mLdesc.empty()) keptDescriptors.push_back(mLdesc.row(static_cast<int>(i)));
                }
            }
            mvKeylinesUn.swap(keptLines);
            mvKeyLineFunctions.swap(keptLineFunctions);
            mLdesc = keptDescriptors;
        }

#ifdef REMOVE_DYNAMIC_OBJECT
        EvaluateDynamicProbabilities(imGray, imGrayPre, imDepthScaled, staticMask);
        imGrayPre = imGray.clone();
#else
        mvDynamicProbability.assign(mvKeys.size(), 0.f);
#endif

        N = mvKeys.size();
        NL = mvKeylinesUn.size();

        mnPlaneNum = mvPlanePoints.size();
        mvpMapPlanes = vector<MapPlane *>(mnPlaneNum, static_cast<MapPlane *>(nullptr));
        mvpParallelPlanes = vector<MapPlane *>(mnPlaneNum, static_cast<MapPlane *>(nullptr));
        mvpVerticalPlanes = vector<MapPlane *>(mnPlaneNum, static_cast<MapPlane *>(nullptr));
        mvbPlaneOutlier = vector<bool>(mnPlaneNum, false);
        mvbVerPlaneOutlier = vector<bool>(mnPlaneNum, false);
        mvbParPlaneOutlier = vector<bool>(mnPlaneNum, false);

        GetLineDepth(imDepthScaled);

        if (mvKeys.empty())
            return;

        UndistortKeyPoints();

        ComputeStereoFromRGBD(imDepthScaled);

        mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
        mvpMapLines = vector<MapLine *>(NL, static_cast<MapLine *>(NULL));
        mvbOutlier = vector<bool>(N, false);
        mvbLineOutlier = vector<bool>(NL, false);


        // This is done only for the first Frame (or after a change in the calibration)
        if (mbInitialComputations) {
            ComputeImageBounds(imGray);

            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;

            mbInitialComputations = false;
        }

        mb = mbf / fx;

        AssignFeaturesToGrid();
    }

    void Frame::AssignFeaturesToGrid() {
        int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
        for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
            for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
                mGrid[i][j].reserve(nReserve);

        for (int i = 0; i < N; i++) {
            const cv::KeyPoint &kp = mvKeysUn[i];

            int nGridPosX, nGridPosY;
            if (PosInGrid(kp, nGridPosX, nGridPosY))
                mGrid[nGridPosX][nGridPosY].push_back(i);
        }
    }

    void Frame::ExtractLSD(const cv::Mat &im) {
        mpLineSegment->ExtractLineSegment(im, mvKeylinesUn, mLdesc, mvKeyLineFunctions);

    }

    void Frame::ExtractORB(const cv::Mat &im, const cv::Mat &staticMask) {
        (*mpORBextractorLeft)(im, staticMask, mvKeys, mDescriptors);
    }

    cv::Mat Frame::BuildStaticMask(const cv::Size &size, const cv::Mat &depth) const {
        cv::Mat staticMask(size, CV_8UC1, cv::Scalar(255));
        if(mpTracker == NULL || mpTracker->yoloBoundingBoxList.empty()) return staticMask;

        cv::Mat membership = planeDetector.plane_filter.membershipImg;
        std::map<int, int> planeArea;
        if(!membership.empty()) {
            for(int y = 0; y < membership.rows; ++y)
                for(int x = 0; x < membership.cols; ++x) {
                    const int id = membership.at<int>(y,x);
                    if(id >= 0) ++planeArea[id];
                }
        }
        const int rigidPlaneMinArea = membership.empty() ? 0 :
            std::max(64, static_cast<int>(membership.total() * 0.02));

        const std::set<std::string> dynamicClasses = {
            "person", "cat", "dog", "bird", "horse", "sheep", "cow"
        };
        for(const YoloBoundingBox &box : mpTracker->yoloBoundingBoxList) {
            if(dynamicClasses.count(box.GetLabel()) == 0 || !box.isDynamic) continue;
            cv::Rect rect = box.GetRect();
            rect &= cv::Rect(0, 0, size.width, size.height);
            if(rect.empty()) continue;
            staticMask(rect).setTo(cv::Scalar(0));

            if(membership.empty()) continue;
            for(int y = rect.y; y < rect.y + rect.height; ++y) {
                for(int x = rect.x; x < rect.x + rect.width; ++x) {
                    if(depth.empty() || !std::isfinite(depth.at<float>(y,x)) || depth.at<float>(y,x) <= 0.f) continue;
                    const int mx = std::min(membership.cols - 1, x * membership.cols / size.width);
                    const int my = std::min(membership.rows - 1, y * membership.rows / size.height);
                    const int planeId = membership.at<int>(my,mx);
                    if(planeId >= 0 && planeArea[planeId] >= rigidPlaneMinArea)
                        staticMask.at<uchar>(y,x) = 255;
                }
            }
        }
        return staticMask;
    }

    void Frame::EvaluateDynamicProbabilities(const cv::Mat &currentGray,
                                              const cv::Mat &previousGray,
                                              const cv::Mat &depth,
                                              const cv::Mat &immunityMask) {
        const float semanticPrior = 0.55f;
        const float shadowPrior = 0.80f;
        const float rejectThreshold = 0.75f;
        mvDynamicProbability.assign(mvKeys.size(), 0.f);

        std::vector<int> suspectIndices;
        std::vector<cv::Point2f> currentPoints;
        for(size_t i = 0; i < mvKeys.size(); ++i) {
            const int x = cvRound(mvKeys[i].pt.x), y = cvRound(mvKeys[i].pt.y);
            if(x >= 0 && x < immunityMask.cols && y >= 0 && y < immunityMask.rows &&
               immunityMask.at<uchar>(y,x) == 0) {
                mvDynamicProbability[i] = semanticPrior;
                suspectIndices.push_back(static_cast<int>(i));
                currentPoints.push_back(mvKeys[i].pt);
            }
        }

        if(!previousGray.empty() && currentPoints.size() >= 8) {
            std::vector<cv::Point2f> previousPoints;
            std::vector<uchar> tracked;
            std::vector<float> errors;
            cv::calcOpticalFlowPyrLK(currentGray, previousGray, currentPoints,
                                     previousPoints, tracked, errors);
            std::vector<cv::Point2f> validCurrent, validPrevious;
            std::vector<int> validIndices;
            for(size_t i = 0; i < tracked.size(); ++i) if(tracked[i]) {
                validCurrent.push_back(currentPoints[i]);
                validPrevious.push_back(previousPoints[i]);
                validIndices.push_back(suspectIndices[i]);
            }
            if(validCurrent.size() >= 8) {
                cv::Mat F = cv::findFundamentalMat(validPrevious, validCurrent,
                                                    cv::FM_RANSAC, 1.5, 0.99);
                if(F.rows == 3 && F.cols == 3) {
                    F.convertTo(F, CV_64F);
                    for(size_t k = 0; k < validCurrent.size(); ++k) {
                        const cv::Matx31d x1(validPrevious[k].x, validPrevious[k].y, 1.0);
                        const cv::Matx31d x2(validCurrent[k].x, validCurrent[k].y, 1.0);
                        cv::Matx33d fm(F.at<double>(0,0), F.at<double>(0,1), F.at<double>(0,2),
                                       F.at<double>(1,0), F.at<double>(1,1), F.at<double>(1,2),
                                       F.at<double>(2,0), F.at<double>(2,1), F.at<double>(2,2));
                        const cv::Matx31d fx1 = fm * x1;
                        const cv::Matx31d ftx2 = fm.t() * x2;
                        const double numerator = x2.dot(fx1);
                        const double denominator = fx1(0)*fx1(0) + fx1(1)*fx1(1) +
                                                   ftx2(0)*ftx2(0) + ftx2(1)*ftx2(1) + 1e-9;
                        const double sampson = numerator * numerator / denominator;
                        const float pGeo = static_cast<float>(1.0 - std::exp(-sampson / 2.0));
                        const int idx = validIndices[k];
                        mvDynamicProbability[idx] = std::max(mvDynamicProbability[idx], pGeo);
                    }
                }
            }
        }

        // Project a conservative footprint from the lower part of each dynamic
        // detection. Only pixels supported by an extracted plane receive the
        // shadow prior, avoiding rejection of arbitrary image regions.
        const cv::Mat membership = planeDetector.plane_filter.membershipImg;
        for(size_t i = 0; i < mvKeys.size() && !membership.empty(); ++i) {
            const cv::Point2f p = mvKeys[i].pt;
            const int mx = std::min(membership.cols - 1, std::max(0, cvRound(p.x) * membership.cols / currentGray.cols));
            const int my = std::min(membership.rows - 1, std::max(0, cvRound(p.y) * membership.rows / currentGray.rows));
            if(membership.at<int>(my,mx) < 0) continue;
            for(const YoloBoundingBox &box : mpTracker->yoloBoundingBoxList) {
                if(!box.isDynamic) continue;
                const cv::Rect2f r = box.GetRect();
                const cv::Rect2f shadow(r.x - 0.15f*r.width, r.y + 0.75f*r.height,
                                        1.30f*r.width, 0.60f*r.height);
                if(shadow.contains(p))
                    mvDynamicProbability[i] = std::max(mvDynamicProbability[i], shadowPrior);
            }
        }

        std::vector<cv::KeyPoint> keptKeys;
        std::vector<float> keptProbabilities;
        cv::Mat keptDescriptors;
        for(size_t i = 0; i < mvKeys.size(); ++i) {
            if(mvDynamicProbability[i] >= rejectThreshold) {
                mvKeysDynamic.push_back(mvKeys[i]);
                if(!mDescriptors.empty()) mDescriptorsDynamic.push_back(mDescriptors.row(static_cast<int>(i)));
            } else {
                keptKeys.push_back(mvKeys[i]);
                keptProbabilities.push_back(mvDynamicProbability[i]);
                if(!mDescriptors.empty()) keptDescriptors.push_back(mDescriptors.row(static_cast<int>(i)));
            }
        }
        mvKeys.swap(keptKeys);
        mvDynamicProbability.swap(keptProbabilities);
        mDescriptors = keptDescriptors;
    }

    void Frame::GetLineDepth(const cv::Mat &imDepth) {
        mvDepthLine = std::vector<std::pair<float, float>>(mvKeylinesUn.size(), make_pair(-1.0f, -1.0f));

        for (int i = 0; i < mvKeylinesUn.size(); ++i) {
            mvDepthLine[i] = std::make_pair(imDepth.at<float>(mvKeylinesUn[i].startPointY, mvKeylinesUn[i].startPointX),
                                            imDepth.at<float>(mvKeylinesUn[i].endPointY, mvKeylinesUn[i].endPointX));
        }
    }

    void Frame::SetPose(cv::Mat Tcw) {
        mTcw = Tcw.clone();
        UpdatePoseMatrices();
    }

    void Frame::UpdatePoseMatrices() {
        mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
        mRwc = mRcw.t();
        mtcw = mTcw.rowRange(0, 3).col(3);
        mOw = -mRcw.t() * mtcw;

        mTwc = cv::Mat::eye(4, 4, mTcw.type());
        mRwc.copyTo(mTwc.rowRange(0, 3).colRange(0, 3));
        mOw.copyTo(mTwc.rowRange(0, 3).col(3));
    }
    //ovd add
void Frame::EpiloarWithFundMatrix(const cv::Mat &ImGrayPre, const cv::Mat &ImGray, const cv::Mat &imDepthScaled)
{
    Curpoint.clear();
    Prepoint.clear();
    epiErr.clear();
    state.clear();
    Err.clear();


    for(auto kp=mvKeys.begin(); kp!=mvKeys.end(); kp++)
    {
        //unique_lock<mutex> lock(mutex);
        bool flag = true;

        int i = kp - mvKeys.begin();
        for(auto it=mpTracker->yoloBoundingBoxList.begin(); it!=mpTracker->yoloBoundingBoxList.end(); it++)
        {
            if(isInBox(*kp,*it,true))
                it->key_in_box.emplace_back(kp->pt);
            //if(it->GetLabel()=="person"&&it->proStatic<0.96)
            if(it->GetLabel()=="person" && isInBox(*kp,*it,true))
            {
                flag = false;
                break;
            }
        }
        if(flag)
            Curpoint.emplace_back(kp->pt);
       
        
    }

    //估计F
    cv::calcOpticalFlowPyrLK(ImGray,ImGrayPre,Curpoint,Prepoint,state,Err, cv::Size(21, 21), 3);

    // //将跟踪误差小的用来估计基础矩阵，提高基础矩阵计算速度
    std::vector<cv::Point2f> PrepointT = Prepoint;
    std::vector<cv::Point2f> CurpointT = Curpoint;
   
    // std::vector<float> flowErr = Err;
    // sort(flowErr.begin(),flowErr.end());
    // float th = flowErr[0.8*flowErr.size()];
    // std::vector<bool> isErase(Curpoint.size(),true);
   
    // for(auto err=Err.begin(); err!=Err.end(); ++err)
    // {
    //     int i = err - Err.begin();
    //     if((*err)>th)
    //     {
    //         isErase[i] = false;
    //     }
    // }
    // reduceVector(Prepoint,isErase);
    // reduceVector(Curpoint,isErase);
    int N = Curpoint.size();
   // cal_f_num.emplace_back(N);
    cv::Mat FundMatrix;
    if(N>8)
    {
        
        if(Curpoint.size()<15)
        {

            FundMatrix = cv::findFundamentalMat(CurpointT,PrepointT,
                                                cv::FM_RANSAC,0.1,0.99,status);

            //计算极线距离
            //CalcEpiDistChi2(mvKeys,PrepointT,FundMatrix,state);
        }
        else
        {

            FundMatrix = cv::findFundamentalMat(Curpoint,Prepoint,
                                                cv::FM_RANSAC,0.1,0.99,status);

            
            //CalcEpiDistChi2(mvKeys,PrepointT,FundMatrix,state);
           
        }
    }
    if(!mpTracker->yoloBoundingBoxList.empty())
    {
        for(auto it = mpTracker->yoloBoundingBoxList.begin(); it -mpTracker->yoloBoundingBoxList.begin()<mpTracker->yoloBoundingBoxList.size(); it++)
        {
            if(it->key_in_box.size() && it->GetLabel()=="person")
            {
                pre_key_box.clear();
                state_box.clear();
                Err_box.clear();
                
                cv::calcOpticalFlowPyrLK(ImGray,ImGrayPre,it->key_in_box, pre_key_box, state_box,Err_box, cv::Size(21, 21), 3);
                
                key_box_dep(it->key_in_box,pre_key_box,FundMatrix, imDepthScaled, *it);
            }
        }
    }
    // if(!mpTracker->yoloBoundingBoxList.empty())
    // {
    //     for(auto it = mpTracker->yoloBoundingBoxList.begin(); it -mpTracker->yoloBoundingBoxList.begin()<mpTracker->yoloBoundingBoxList.size(); it++)
    //     {
    //         cur_box.clear();
    //         pre_box.clear();
    //         epiErr_box.clear();
    //         state_box.clear();
    //         Err_box.clear();
    //         cv::Rect2f rect_object = it->GetRect();
    //         cv::Point2f p1 = rect_object.tl();
    //         cv::Point2f p2 = rect_object.br();
    //         int x1 = (int) p1.x ;
    //         int y1 = (int) p1.y ;
    //         int x2 = (int) p2.x ;
    //         int y2 = (int) p2.y ;
            
    //         for(int i=x1; i<x2; i+=6)
    //         {
    //             for(int j=y1; j<y2; j+=6)
    //             {
    //                 if(i>=0 && i<imDepthScaled.cols && j>=0 && j<imDepthScaled.rows)
    //                 {
    //                     //float ni = static_cast<float>(i);
    //                     //float nj = static_cast<float>(j);
    //                     cv::Point2f a;
    //                     a.x = i;
    //                     a.y = j;
    //                     cur_box.emplace_back(a);
    //                 }
            
    //             }
    //         } 
    //         cv::calcOpticalFlowPyrLK(ImGray,ImGrayPre,cur_box,pre_box,state_box,Err_box, cv::Size(21, 21), 3);
    //         CalcEpiDistChi2(cur_box,pre_box,FundMatrix,state_box);
    //         it->epiErr_box.swap(epiErr_box);
    //     }
            
    // }

}
#include <iostream>
#include <fstream>

void Frame::CalcEpiDistChi2(const std::vector<cv::Point2f> &kp1, const std::vector<cv::Point2f> &kp2, cv::Mat &F, std::vector<uchar> state)
{
        int N=kp1.size();
        for(size_t i=0; i<N; i++)
        {
            double A1 = F.at<double>(0, 0)*kp1[i].x + F.at<double>(0, 1)*kp1[i].y + F.at<double>(0, 2);
            double B1 = F.at<double>(1, 0)*kp1[i].x + F.at<double>(1, 1)*kp1[i].y + F.at<double>(1, 2);
            double C1 = F.at<double>(2, 0)*kp1[i].x + F.at<double>(2, 1)*kp1[i].y + F.at<double>(2, 2);
            double dis1 = fabs(A1*kp2[i].x + B1*kp2[i].y + C1) / sqrt(A1*A1 + B1*B1);

            double A2 = F.at<double>(0, 0)*kp2[i].x + F.at<double>(1, 0)*kp2[i].y + F.at<double>(2, 0);
            double B2 = F.at<double>(0, 1)*kp2[i].x + F.at<double>(1, 1)*kp2[i].y + F.at<double>(2, 1);
            double C2 = F.at<double>(0, 2)*kp2[i].x + F.at<double>(1, 2)*kp2[i].y + F.at<double>(2, 2);
            double dis2 = fabs(A2*kp1[i].x + B2*kp1[i].y + C2) / sqrt(A2*A2 + B2*B2);


            double dis = (dis1 + dis2)/2.0;
            if(isnan(dis) || isinf(dis))
                dis = 999;
          
            EpiErr_box.emplace_back(dis);
            // 定义卡方分布变量
            boost::math::chi_squared_distribution<> dist(2);
            //cout<< "CalcEpiDistChi2 z正常: "<<i<<" value : "<<dis<<" 总共: "<<N<<endl;
            epiErr_box.emplace_back(2*boost::math::pdf(dist, dis*dis));
            
        }

}
void Frame::key_box_dep(const std::vector<cv::Point2f> &kp1, const std::vector<cv::Point2f> &kp2, 
        cv::Mat &F, const cv::Mat imDepthScaled, YoloBoundingBox &it)
{
    int N=kp1.size();
    vector<float> tmp;
    float sum = 0;
    for(size_t i=0; i<N; i++)
    {
        double A1 = F.at<double>(0, 0)*kp1[i].x + F.at<double>(0, 1)*kp1[i].y + F.at<double>(0, 2);
        double B1 = F.at<double>(1, 0)*kp1[i].x + F.at<double>(1, 1)*kp1[i].y + F.at<double>(1, 2);
        double C1 = F.at<double>(2, 0)*kp1[i].x + F.at<double>(2, 1)*kp1[i].y + F.at<double>(2, 2);
        double dis1 = fabs(A1*kp2[i].x + B1*kp2[i].y + C1) / sqrt(A1*A1 + B1*B1);

        double A2 = F.at<double>(0, 0)*kp2[i].x + F.at<double>(1, 0)*kp2[i].y + F.at<double>(2, 0);
        double B2 = F.at<double>(0, 1)*kp2[i].x + F.at<double>(1, 1)*kp2[i].y + F.at<double>(2, 1);
        double C2 = F.at<double>(0, 2)*kp2[i].x + F.at<double>(1, 2)*kp2[i].y + F.at<double>(2, 2);
        double dis2 = fabs(A2*kp1[i].x + B2*kp1[i].y + C2) / sqrt(A2*A2 + B2*B2);
        double dis = (dis1 + dis2)/2.0;
        //cout<<dis<<" ";
        if(!isnan(dis) || !isinf(dis))
        {
            //it.key_in_box.emplace_back(*kp1[i]);
            if(imDepthScaled.at<float>(kp1[i].y, kp1[i].x) >0)
            {
                boost::math::chi_squared_distribution<> dist(2);
                float dis= imDepthScaled.at<float>(kp1[i].y, kp1[i].x);
                
                //tmp.emplace_back(2*boost::math::pdf(dist, dis*dis));
                tmp.emplace_back(dis);
            }
            
        }
        
    }
    
    sort(tmp.begin(),tmp.end());
    int M = tmp.size();
    //tmp.erase(tmp.begin()+0.7*M,tmp.end());
    for(auto p:tmp)
        sum += p;
    it.key_dep_average = sum/tmp.size();
    sum = 0.0;
    for(auto d:tmp)
    {
        sum += (d - it.key_dep_average)*(d - it.key_dep_average);
    }
    it.key_dep_stdcov = std::sqrt(sum/(tmp.size()));
}
void Frame::CalcEpiDistChi2(const std::vector<cv::KeyPoint> &kp1, const std::vector<cv::Point2f> &kp2, cv::Mat &F, std::vector<uchar> state)
{
        int N=kp1.size();
        for(size_t i=0; i<N; i++)
        {
            double A1 = F.at<double>(0, 0)*kp1[i].pt.x + F.at<double>(0, 1)*kp1[i].pt.y + F.at<double>(0, 2);
            double B1 = F.at<double>(1, 0)*kp1[i].pt.x + F.at<double>(1, 1)*kp1[i].pt.y + F.at<double>(1, 2);
            double C1 = F.at<double>(2, 0)*kp1[i].pt.x + F.at<double>(2, 1)*kp1[i].pt.y + F.at<double>(2, 2);
            double dis1 = fabs(A1*kp2[i].x + B1*kp2[i].y + C1) / sqrt(A1*A1 + B1*B1);

            double A2 = F.at<double>(0, 0)*kp2[i].x + F.at<double>(1, 0)*kp2[i].y + F.at<double>(2, 0);
            double B2 = F.at<double>(0, 1)*kp2[i].x + F.at<double>(1, 1)*kp2[i].y + F.at<double>(2, 1);
            double C2 = F.at<double>(0, 2)*kp2[i].x + F.at<double>(1, 2)*kp2[i].y + F.at<double>(2, 2);
            double dis2 = fabs(A2*kp1[i].pt.x + B2*kp1[i].pt.y + C2) / sqrt(A2*A2 + B2*B2);


            double dis = (dis1 + dis2)/2.0;
            if(isnan(dis) || isinf(dis))
                dis = 999;
            
            EpiErr.emplace_back(dis);
            // 定义卡方分布变量
            boost::math::chi_squared_distribution<> dist(2);
            
            epiErr.emplace_back(2*boost::math::pdf(dist, dis*dis));
            
        }
        
        
        // for(size_t i=0; i<NL; i++)
        // {
        //     //line_start
        //     double A1_s = F.at<double>(0, 0)*s1[i].x + F.at<double>(0, 1)*s1[i].y + F.at<double>(0, 2);
        //     double B1_s = F.at<double>(1, 0)*s1[i].x + F.at<double>(1, 1)*s1[i].y + F.at<double>(1, 2);
        //     double C1_s = F.at<double>(2, 0)*s1[i].x + F.at<double>(2, 1)*s1[i].y + F.at<double>(2, 2);
        //     double dis1 = fabs(A1_s*s2[i].x + B1_s*s2[i].y + C1_s) / sqrt(A1_s*A1_s + B1_s*B1_s);

        //     // double A2_s = F.at<double>(0, 0)*s2[i].x + F.at<double>(1, 0)*s2[i].y + F.at<double>(2, 0);
        //     // double B2_s = F.at<double>(0, 1)*s2[i].x + F.at<double>(1, 1)*s2[i].y + F.at<double>(2, 1);
        //     // double C2_s = F.at<double>(0, 2)*s2[i].x + F.at<double>(1, 2)*s2[i].y + F.at<double>(2, 2);
        //     // double dis2 = fabs(A2_s*s1[i].x + B2_s*s1[i].y + C2_s) / sqrt(A2_s*A2_s + B2_s*B2_s);

        //     double dis = dis1;
        //     if(isnan(dis) || isinf(dis))
        //         dis = 9999;
        //     // 定义卡方分布变量
        //     boost::math::chi_squared_distribution<> dist(2);
        //     epiErr_line_start.emplace_back(2*boost::math::pdf(dist, dis*dis));

        //     //line_end 
        //     double A1_e = F.at<double>(0, 0)*e1[i].x + F.at<double>(0, 1)*e1[i].y + F.at<double>(0, 2);
        //     double B1_e = F.at<double>(1, 0)*e1[i].x + F.at<double>(1, 1)*e1[i].y + F.at<double>(1, 2);
        //     double C1_e = F.at<double>(2, 0)*e1[i].x + F.at<double>(2, 1)*e1[i].y + F.at<double>(2, 2);
        //     double dis1e = fabs(A1_e*e2[i].x + B1_e*e2[i].y + C1_e) / sqrt(A1_e*A1_e + B1_e*B1_e);

        //     // double A2_e = F.at<double>(0, 0)*e2[i].x + F.at<double>(1, 0)*e2[i].y + F.at<double>(2, 0);
        //     // double B2_e = F.at<double>(0, 1)*e2[i].x + F.at<double>(1, 1)*e2[i].y + F.at<double>(2, 1);
        //     // double C2_e = F.at<double>(0, 2)*e2[i].x + F.at<double>(1, 2)*e2[i].y + F.at<double>(2, 2);
        //     // double dis2e = fabs(A2_e*e1[i].x + B2_e*e1[i].y + C2_e) / sqrt(A2_e*A2_e + B2_e*B2_e);

        //     dis = dis1e;
        //     if(isnan(dis) || isinf(dis))
        //         dis = 9999;
        //     // 定义卡方分布变量
        //     //boost::math::chi_squared_distribution<> dist(2);
        //     epiErr_line_end.emplace_back(2*boost::math::pdf(dist, dis*dis));
        //}
        
//    });
}
void Frame::RemoveMovingPointsWithEpiAndYOLO(std::vector<double> epiErr, 
                                             const cv::Mat &ImGray, const cv::Mat &ImDepth)
{
    if(epiErr.empty())
        return;
    std::vector<cv::KeyPoint> staticPoints;
    std::vector<double> epiErrStatic;
    std::vector<cv::KeyPoint> _mvKeys;
    cv::Mat _mDescriptors;
    
    //line
    std::vector<cv::line_descriptor::KeyLine> _line;
    cv::Mat _lbd;
    
    std::vector<Eigen::Vector3d> mvKeyLineFunctions_;
    vector<Vector6d > mvLines3D_;
    std::vector<float>  mvDepthLine_;
    //没有检测到物体，仅做极线匹配
    if(mpTracker->yoloBoundingBoxList.empty())
    {
        for(int i=0; i<mvKeys.size(); i++)
        {
            if(EpiErr[i] < 2.0)
            {
                _mvKeys.emplace_back(mvKeys[i]);
                _mDescriptors.push_back(mDescriptors.row(i));
            }
            else
            {
                mvKeysDynamic.emplace_back(mvKeys[i]);
                mDescriptorsDynamic.push_back(mDescriptors.row(i));
            }
        }
        mvKeys.swap(_mvKeys);
        mDescriptors = _mDescriptors.clone();
        
        //line
    //    for(int i=0; i<mvKeylinesUn.size(); i++)
    //     {
    //         if(epiErr_line_start[i] < 5.2 && EpiErr_line_end[i] < 5.2) 
    //         {
    //             _line.emplace_back(mvKeylinesUn[i]);
    //             _lbd.push_back(mLdesc.row(i));
    //             mvKeyLineFunctions_.push_back(mvKeyLineFunctions[i]);
       
    //         }
    //         else
    //         {
    //             mvLineDynamic.emplace_back(mvKeylinesUn[i]);CalcEpiDistChi2(cur_box,pre_box,FundMatrix,state);
                
    //             mlbdDynamic.push_back(mLdesc.row(i));
    //         }
    //     }
    //     mvKeylinesUn.swap(_line);
    //     mLdesc = _lbd.clone(); 
    //     mvKeyLineFunctions.swap(mvKeyLineFunctions_);
  
    }
    else
    {
        //遍历所有点
        for(int i=0; i<mvKeys.size(); i++)
        {
            //遍历所有框
            for(int j=0; j<mpTracker->yoloBoundingBoxList.size(); j++)
            {
                //判断是否在框内
                if(isInBox(mvKeys[i],mpTracker->yoloBoundingBoxList[j]))
                {
                    mpTracker->yoloBoundingBoxList[j].KeyPointsInBox.emplace_back(make_pair(i,mvKeys[i]));
                    mpTracker->yoloBoundingBoxList[j].epiErr.emplace_back(epiErr[i]);   //分布
                }
                
//                else
//                {
//                    staticPoints.emplace_back(mvKeys[i]);
//                    epiErrStatic.emplace_back(epiErr[i]);
//                }
            }
        }
    //    for(int i=0; i<mvKeylinesUn.size(); i++)
    //     {
    //         //遍历所有框
    //         for(int j=0; j<mpTracker->yoloBoundingBoxList.size(); j++)
    //         {
    //             //判断是否在框内
    //             //line
    //             if(lineisInBox(mvKeylinesUn[i],mpTracker->yoloBoundingBoxList[j])) 
    //             {
    //                 //将特征线放入框中
    //                 mpTracker->yoloBoundingBoxList[j].KeyLinesInBox.emplace_back(make_pair(i,mvKeylinesUn[i]));
    //                 mpTracker->yoloBoundingBoxList[j].epiErr_line.emplace_back(max(epiErr_line_start[i],epiErr_line_end[i]));
    //             }
    //         }
    //     } 
        //计算每个框的动静属性
        for(auto it=mpTracker->yoloBoundingBoxList.begin(); it!=mpTracker->yoloBoundingBoxList.end(); ++it)
        {
            int M = it->epiErr_box.size();
            int M1 = 0.1*M;
            int M2 = 0.2*M;
            int M3 = 0.3*M;
            if(M == 0)
                continue;
            sort(it->epiErr_box.begin(),it->epiErr_box.end());
            double proStatic = (it->epiErr_box[M1]+it->epiErr_box[M2]+it->epiErr_box[M3])/3.0;

           
           
//            cout<<it.GetLabel()<<": "<<proStatic<<endl;
//            cout<<mpTracker->yoloBoundingBoxList.size()<<endl;
            if(proStatic<1.0)
            {
                if(it->GetLabel() == "person" && proStatic<0.9)
                {
                    
                    it->isDynamic = true;
                    it->proStatic = proStatic;
                }
                else if(it->GetLabel() != "person" && proStatic<0.8)
                {
                    it->isDynamic = true;
                    it->proStatic = proStatic;
                }
                else
                {
                    it->isDynamic = false;
                    it->proStatic = proStatic;
                }
            }
        }
    }
}
bool Frame::isInBox(cv::KeyPoint &kp,YoloBoundingBox &Box,bool extend)
{
    cv::Rect2f tmpRect = Box.GetRect();
    if(!extend) {
        if (kp.pt.x > tmpRect.tl().x - 10
            && kp.pt.y > tmpRect.tl().y - 10
            && kp.pt.x < tmpRect.br().x + 10
            && kp.pt.y < tmpRect.br().y + 10) {
            return true;
        }
    }
    else {
        if (kp.pt.x > tmpRect.tl().x 
            && kp.pt.y > tmpRect.tl().y 
            && kp.pt.x < tmpRect.br().x 
            && kp.pt.y < tmpRect.br().y ) {
            return true;
        }
    }
    return false;
}
bool Frame::lineisInBox(cv::line_descriptor::KeyLine &mvkeyline,YoloBoundingBox &Box,bool extend )
{
     cv::Rect2f tmpRect = Box.GetRect();
    if(!extend) {
        if (mvkeyline.startPointX > tmpRect.tl().x 
            && mvkeyline.startPointY > tmpRect.tl().y 
            && mvkeyline.startPointX < tmpRect.br().x 
            && mvkeyline.startPointY < tmpRect.br().y ) {
            return true;
        }
    }
    else {
        if (mvkeyline.startPointX > tmpRect.tl().x - 0.2 * Box.width
            && mvkeyline.startPointY > tmpRect.tl().y - 10
            && mvkeyline.startPointX < tmpRect.br().x + 0.2 * Box.width
            && mvkeyline.startPointY < tmpRect.br().y + 10) {
            return true;
        }
    }
    return false;
}
void Frame::reduceVector(std::vector<cv::Point2f> &v, std::vector<bool> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}
void Frame::reduceLineVector(std::vector<cv::Point2f> &s, std::vector<bool> status_s, std::vector<cv::Point2f> &e, std::vector<bool> status_e)
{
    int j = 0;
    for (int i = 0; i < int(s.size()); i++)
        if (status_s[i] && status_e[i])
        {
            s[j] = s[i];
            e[j] = e[i];
            j++;
        }
    s.resize(j);
    e.resize(j);
}
void Frame::RemoveMovingKeyPoints(const cv::Mat &ImGray, const cv::Mat &imDepth)
{
    // if(epiErr.empty())
    //     return;
    std::vector<cv::KeyPoint> _mvKeys;
    cv::Mat _mDescriptors;

    std::vector<Eigen::Vector3d> mvKeyLineFunctions_;
    vector<Vector6d > mvLines3D_;
    std::vector<float>  mvDepthLine_;
    std::vector<cv::line_descriptor::KeyLine> _mvKeylinesUn;
    cv::Mat _lbd;


//     for(auto it=mpTracker->yoloBoundingBoxList.begin(); it!=mpTracker->yoloBoundingBoxList.end(); ++it)
//     {
//         int BoxSize = it->KeyPointsInBox.size();
//         if(BoxSize == 0)
//             continue;
//         //将框内的疑似动态点放入容器
//         for(int i=0; i<BoxSize; ++i)
//         {
//             it->mvKeysDynam.emplace_back(imDepth.at<float>(mvKeys[it->KeyPointsInBox[i].first].pt.y,
//                                                            mvKeys[it->KeyPointsInBox[i].first].pt.x));
//         }
//         //计算当前框内点深度的平均值和标准差
//         int M = it->mvKeysDynam.size();
//         sort(it->mvKeysDynam.begin(),it->mvKeysDynam.end());
//         it->mvKeysDynam.erase(it->mvKeysDynam.begin()+0.8*M,it->mvKeysDynam.end());
//         M = it->mvKeysDynam.size();
//         it->average = std::accumulate(it->mvKeysDynam.begin(),it->mvKeysDynam.end(),0.0)/M;
// //        it->average = (it->mvKeysDynam[0.3*M]+it->mvKeysDynam[0.5*M]+it->mvKeysDynam[0.7*M])/3.0;
//         float sum = 0.0;
//         for(auto d:it->mvKeysDynam)
//             sum += (d-it->average)*(d-it->average);

//         it->stdcov = sum/(M);
//         //cout<<"KeyPointsInBox size "<<it->KeyPointsInBox.size()<<endl;
//     }
   
    // 对所有点进行判断
    for(auto kp=mvKeys.begin(); kp!=mvKeys.end(); ++kp)
    {
        //if(isInBox(*kp,*it,true))
        //bool flag = true;
        bool Isstatic = false;
        int i = kp - mvKeys.begin();
        pair<bool, bool> flag(true, true);
        bool ren = false;
        
        //cout<<"normal ";
        //mpTracker->show_semantic_mask.at<unsigned short>(kp->pt.y, kp->pt.x)==0 &&
    //     if( chi * chi >= 1.9)
    //     {
    // //            epiErr2.emplace_back(epiErr[i]);
    //         _mvKeys.emplace_back(mvKeys[i]);
    //         _mDescriptors.push_back(mDescriptors.row(i));
    //     }
    //     else
    //     {
        for(auto it=mpTracker->yoloBoundingBoxList.begin(); it!=mpTracker->yoloBoundingBoxList.end(); ++it)
        {
            if(isInBox(*kp,*it,true))
            {
                if(it->GetLabel()=="person")
                {
                    ren = true;
                    
                    float chi = (imDepth.at<float>(kp->pt.y,kp->pt.x) - it->key_dep_average) / it->key_dep_stdcov;
                    if(chi * chi < 1.9)
                    {
                        flag.first = false;
                        break;
                    }
                        
                }
                
            }
            // if(ren)
            // {
            //     if(flag.second)
            //     {
            //         _mvKeys.emplace_back(mvKeys[i]);
            //         _mDescriptors.push_back(mDescriptors.row(i));
            //     }
            //     else
            //     {
            //         mvKeysDynamic.emplace_back(mvKeys[i]);
            //         mDescriptorsDynamic.push_back(mDescriptors.row(i)); 
            //     }
            // }
            // else
            // {
            //     if(flag.first && flag.second)
            //     {
            //         _mvKeys.emplace_back(mvKeys[i]);
            //         _mDescriptors.push_back(mDescriptors.row(i));
            //     }
            //     else
            //     {
            //         mvKeysDynamic.emplace_back(mvKeys[i]);
            //         mDescriptorsDynamic.push_back(mDescriptors.row(i)); 
            //     }
            // }   
        



        
        

        // if(mpTracker->semantic_mask.at<unsigned short>(kp->pt.y, kp->pt.x) != 0 )
        //     flag = false;
        
        // for(auto it=mpTracker->yoloBoundingBoxList.begin(); it!=mpTracker->yoloBoundingBoxList.end(); ++it)
        // {
            
        //     //if(it->GetLabel()=="person"&&it->proStatic<0.96)
        //     if(it->GetLabel()=="person")
        //     { 
        //         if(isInBox(*kp,*it,true) )
        //         {
        //             flag = false;
        //             const float invSigma2 = 128 * imDepth.at<float>(kp->pt.y, kp->pt.x) *
        //                                     imDepth.at<float>(kp->pt.y, kp->pt.x) / (fx);
        //             float chi = (imDepth.at<float>(kp->pt.y, kp->pt.x) - it->average) / it->stdcov;

        //             float chi2 = (imDepth.at<float>(kp->pt.y, kp->pt.x) - it->dep_average) / it->dep_stdcov;
        //             //if (chi * chi > 3.2 && EpiErr[i]<0.3) 
        //             if( imDepth.at<float>(kp->pt.y, kp->pt.x)>0)
        //             {
        //                 //EpiErr[i]<0.3 && mpTracker->compen_semantic.at<unsigned short>(kp->pt.y, kp->pt.x)==0 &&
        //                 if(
        //                 mpTracker->show_semantic_mask.at<unsigned short>(kp->pt.y, kp->pt.x)==0 )
        //                     flag = true;
                    

        //             }
                
        //         }
        
        //     }
        // }
        // if(EpiErr[i]>20.0)
        //     flag = false;
        //if(flag || Isstatic)
        
        
        }
        if(flag.first && flag.second)  
        {
            _mvKeys.emplace_back(mvKeys[i]);
            _mDescriptors.push_back(mDescriptors.row(i));
        }
        else
        {
            mvKeysDynamic.emplace_back(mvKeys[i]);
            mDescriptorsDynamic.push_back(mDescriptors.row(i)); 
        }
    }
    mvKeys.swap(_mvKeys);
    mDescriptors = _mDescriptors.clone();
    

    for(auto kp=mvKeylinesUn.begin(); kp!=mvKeylinesUn.end(); ++kp)
    {
        bool flag = true;
        bool Isstatic = false;
        int i = kp - mvKeylinesUn.begin();
        for(auto it=mpTracker->yoloBoundingBoxList.begin(); it!=mpTracker->yoloBoundingBoxList.end(); ++it)
        {
            if(it->GetLabel()!="person")
                continue;
            if(lineisInBox(mvKeylinesUn[i],*it))
            {
        //mpTracker->show_semantic_mask.at<unsigned short>(kp->getStartPoint().y, kp->getStartPoint().x) != 0  ||
        //if(mpTracker->compen_semantic.at<unsigned short>(kp->getEndPoint().y, kp->getEndPoint().x) != 0  )
        //    flag = false;
                float chi = (imDepth.at<float>(kp->getEndPoint().y,kp->getEndPoint().x) - it->key_dep_average) / it->key_dep_stdcov;
                if(chi * chi < 1.9)
                    flag = false;
            }
            break;
        }

        if(flag)
        {
            _mvKeylinesUn.emplace_back(mvKeylinesUn[i]);
            _lbd.push_back(mLdesc.row(i));
            mvKeyLineFunctions_.push_back(mvKeyLineFunctions[i]);
        }
        else
        {
//            epiErr2.emplace_back(epiErr[i]);
            mvLineDynamic.emplace_back(mvKeylinesUn[i]);
            mlbdDynamic.push_back(mLdesc.row(i));
        }
    }
    mvKeylinesUn.swap(_mvKeylinesUn);
    mLdesc = _lbd.clone();   

    mvKeyLineFunctions.swap(mvKeyLineFunctions_);  

}
    bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit) {
        pMP->mbTrackInView = false;

        // 3D in absolute coordinates
        cv::Mat P = pMP->GetWorldPos();

        // 3D in camera coordinates
        const cv::Mat Pc = mRcw * P + mtcw;
        const float &PcX = Pc.at<float>(0);
        const float &PcY = Pc.at<float>(1);
        const float &PcZ = Pc.at<float>(2);

        // Check positive depth
        if (PcZ < 0.0f)
            return false;

        // Project in image and check it is not outside
        const float invz = 1.0f / PcZ;
        const float u = fx * PcX * invz + cx;
        const float v = fy * PcY * invz + cy;

        if (u < mnMinX || u > mnMaxX)
            return false;
        if (v < mnMinY || v > mnMaxY)
            return false;

        // Check distance is in the scale invariance region of the MapPoint
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const cv::Mat PO = P - mOw;
        const float dist = cv::norm(PO);

        if (dist < minDistance || dist > maxDistance)
            return false;

        // Check viewing angle
        cv::Mat Pn = pMP->GetNormal();

        const float viewCos = PO.dot(Pn) / dist;

        if (viewCos < viewingCosLimit)
            return false;

        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist, this);

        // Data used by the tracking
        pMP->mbTrackInView = true;
        pMP->mTrackProjX = u;
        pMP->mTrackProjXR = u - mbf * invz;
        pMP->mTrackProjY = v;
        pMP->mnTrackScaleLevel = nPredictedLevel;
        pMP->mTrackViewCos = viewCos;

        return true;
    }

    bool Frame::isInFrustum(MapLine *pML, float viewingCosLimit) {
        pML->mbTrackInView = false;

        Vector6d P = pML->GetWorldPos();

        cv::Mat SP = (Mat_<float>(3, 1) << P(0), P(1), P(2));
        cv::Mat EP = (Mat_<float>(3, 1) << P(3), P(4), P(5));

        const cv::Mat SPc = mRcw * SP + mtcw;
        const float &SPcX = SPc.at<float>(0);
        const float &SPcY = SPc.at<float>(1);
        const float &SPcZ = SPc.at<float>(2);

        const cv::Mat EPc = mRcw * EP + mtcw;
        const float &EPcX = EPc.at<float>(0);
        const float &EPcY = EPc.at<float>(1);
        const float &EPcZ = EPc.at<float>(2);

        if (SPcZ < 0.0f || EPcZ < 0.0f)
            return false;

        const float invz1 = 1.0f / SPcZ;
        const float u1 = fx * SPcX * invz1 + cx;
        const float v1 = fy * SPcY * invz1 + cy;

        if (u1 < mnMinX || u1 > mnMaxX)
            return false;
        if (v1 < mnMinY || v1 > mnMaxY)
            return false;

        const float invz2 = 1.0f / EPcZ;
        const float u2 = fx * EPcX * invz2 + cx;
        const float v2 = fy * EPcY * invz2 + cy;

        if (u2 < mnMinX || u2 > mnMaxX)
            return false;
        if (v2 < mnMinY || v2 > mnMaxY)
            return false;


        const float maxDistance = pML->GetMaxDistanceInvariance();
        const float minDistance = pML->GetMinDistanceInvariance();

        const cv::Mat OM = 0.5 * (SP + EP) - mOw;
        const float dist = cv::norm(OM);

        if (dist < minDistance || dist > maxDistance)
            return false;


        Vector3d Pn = pML->GetNormal();
        cv::Mat pn = (Mat_<float>(3, 1) << Pn(0), Pn(1), Pn(2));
        const float viewCos = OM.dot(pn) / dist;

        if (viewCos < viewingCosLimit)
            return false;

        const int nPredictedLevel = pML->PredictScale(dist, mfLogScaleFactor);

        pML->mbTrackInView = true;
        pML->mTrackProjX1 = u1;
        pML->mTrackProjY1 = v1;
        pML->mTrackProjX2 = u2;
        pML->mTrackProjY2 = v2;
        pML->mnTrackScaleLevel = nPredictedLevel;
        pML->mTrackViewCos = viewCos;

        return true;
    }


    vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel,
                                            const int maxLevel) const {
        vector<size_t> vIndices;
        vIndices.reserve(N);

        const int nMinCellX = max(0, (int) floor((x - mnMinX - r) * mfGridElementWidthInv));
        if (nMinCellX >= FRAME_GRID_COLS)
            return vIndices;

        const int nMaxCellX = min((int) FRAME_GRID_COLS - 1, (int) ceil((x - mnMinX + r) * mfGridElementWidthInv));
        if (nMaxCellX < 0)
            return vIndices;

        const int nMinCellY = max(0, (int) floor((y - mnMinY - r) * mfGridElementHeightInv));
        if (nMinCellY >= FRAME_GRID_ROWS)
            return vIndices;

        const int nMaxCellY = min((int) FRAME_GRID_ROWS - 1, (int) ceil((y - mnMinY + r) * mfGridElementHeightInv));
        if (nMaxCellY < 0)
            return vIndices;

        const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

        for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
            for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
                const vector<size_t> vCell = mGrid[ix][iy];
                if (vCell.empty())
                    continue;

                for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
                    const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                    if (bCheckLevels) {
                        if (kpUn.octave < minLevel)
                            continue;
                        if (maxLevel >= 0)
                            if (kpUn.octave > maxLevel)
                                continue;
                    }

                    const float distx = kpUn.pt.x - x;
                    const float disty = kpUn.pt.y - y;

                    if (fabs(distx) < r && fabs(disty) < r)
                        vIndices.push_back(vCell[j]);
                }
            }
        }

        return vIndices;
    }

    vector<size_t>
    Frame::GetLinesInArea(const float &x1, const float &y1, const float &x2, const float &y2, const float &r,
                          const int minLevel, const int maxLevel) const {
        vector<size_t> vIndices;

        vector<KeyLine> vkl = this->mvKeylinesUn;

        const bool bCheckLevels = (minLevel > 0) || (maxLevel > 0);

        for (size_t i = 0; i < vkl.size(); i++) {
            KeyLine keyline = vkl[i];

            float distance = (0.5 * (x1 + x2) - keyline.pt.x) * (0.5 * (x1 + x2) - keyline.pt.x) +
                             (0.5 * (y1 + y2) - keyline.pt.y) * (0.5 * (y1 + y2) - keyline.pt.y);
            if (distance > r * r)
                continue;

            float slope = (y1 - y2) / (x1 - x2) - keyline.angle;
            if (slope > r * 0.01)
                continue;

            if (bCheckLevels) {
                if (keyline.octave < minLevel)
                    continue;
                if (maxLevel >= 0 && keyline.octave > maxLevel)
                    continue;
            }

            vIndices.push_back(i);
        }

        return vIndices;
    }


    bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY) {
        posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
        posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

        //Keypoint's coordinates are undistorted, which could cause to go out of the image
        if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
            return false;

        return true;
    }


    void Frame::ComputeBoW() {
        if (mBowVec.empty()) {
            vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
            mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
        }
    }

    void Frame::UndistortKeyPoints() {
        if (mDistCoef.at<float>(0) == 0.0) {
            mvKeysUn = mvKeys;
            return;
        }

        // Fill matrix with points
        cv::Mat mat(N, 2, CV_32F);
        for (int i = 0; i < N; i++) {
            mat.at<float>(i, 0) = mvKeys[i].pt.x;
            mat.at<float>(i, 1) = mvKeys[i].pt.y;
        }

        // Undistort points
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
        mat = mat.reshape(1);

        // Fill undistorted keypoint vector
        mvKeysUn.resize(N);
        for (int i = 0; i < N; i++) {
            cv::KeyPoint kp = mvKeys[i];
            kp.pt.x = mat.at<float>(i, 0);
            kp.pt.y = mat.at<float>(i, 1);
            mvKeysUn[i] = kp;
        }
    }

    void Frame::ComputeImageBounds(const cv::Mat &imLeft) {
        if (mDistCoef.at<float>(0) != 0.0) {
            cv::Mat mat(4, 2, CV_32F);
            mat.at<float>(0, 0) = 0.0;
            mat.at<float>(0, 1) = 0.0;
            mat.at<float>(1, 0) = imLeft.cols;
            mat.at<float>(1, 1) = 0.0;
            mat.at<float>(2, 0) = 0.0;
            mat.at<float>(2, 1) = imLeft.rows;
            mat.at<float>(3, 0) = imLeft.cols;
            mat.at<float>(3, 1) = imLeft.rows;

            // Undistort corners
            mat = mat.reshape(2);
            cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
            mat = mat.reshape(1);

            mnMinX = min(mat.at<float>(0, 0), mat.at<float>(2, 0));
            mnMaxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0));
            mnMinY = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
            mnMaxY = max(mat.at<float>(2, 1), mat.at<float>(3, 1));

        } else {
            mnMinX = 0.0f;
            mnMaxX = imLeft.cols;
            mnMinY = 0.0f;
            mnMaxY = imLeft.rows;
        }
    }

    void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth) {
        mvuRight = vector<float>(N, -1);
        mvDepth = vector<float>(N, -1);

        for (int i = 0; i < N; i++) {
            const cv::KeyPoint &kp = mvKeys[i];
            const cv::KeyPoint &kpU = mvKeysUn[i];

            const float &v = kp.pt.y;
            const float &u = kp.pt.x;

            const float d = imDepth.at<float>(v, u);

            if (d > 0) {
                mvDepth[i] = d;
                mvuRight[i] = kpU.pt.x - mbf / d;
            }
        }
    }

    cv::Mat Frame::UnprojectStereo(const int &i) {
        const float z = mvDepth[i];
        if (z > 0) {
            const float u = mvKeysUn[i].pt.x;
            const float v = mvKeysUn[i].pt.y;
            const float x = (u - cx) * z * invfx;
            const float y = (v - cy) * z * invfy;
            cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);
            return mRwc * x3Dc + mOw;
        } else
            return cv::Mat();
    }

    Vector6d Frame::Obtain3DLine(const int &i, const cv::Mat &imDepth) {
        double len = cv::norm(mvKeylinesUn[i].getStartPoint() - mvKeylinesUn[i].getEndPoint());

        vector<cv::Point3d> pts3d;
        // iterate through a line
        double numSmp = (double) min((int) len, 100); //number of line points sampled

        pts3d.reserve(numSmp);

        for (int j = 0; j <= numSmp; ++j) {
            // use nearest neighbor to querry depth value
            // assuming position (0,0) is the top-left corner of image, then the
            // top-left pixel's center would be (0.5,0.5)
            cv::Point2d pt = mvKeylinesUn[i].getStartPoint() * (1 - j / numSmp) +
                             mvKeylinesUn[i].getEndPoint() * (j / numSmp);
            if (pt.x < 0 || pt.y < 0 || pt.x >= imDepth.cols || pt.y >= imDepth.rows) continue;
            int row, col; // nearest pixel for pt
            if ((floor(pt.x) == pt.x) && (floor(pt.y) == pt.y)) { // boundary issue
                col = max(int(pt.x - 1), 0);
                row = max(int(pt.y - 1), 0);
            } else {
                col = int(pt.x);
                row = int(pt.y);
            }

            float d = -1;
            if (imDepth.at<float>(row, col) <= 0.01) { // no depth info
                continue;
            } else {
                d = imDepth.at<float>(row, col);
            }
            cv::Point3d p;

            p.z = d;
            p.x = (col - cx) * p.z * invfx;
            p.y = (row - cy) * p.z * invfy;

            pts3d.push_back(p);

        }

        if (pts3d.size() < 10.0)
            //return static_cast<Vector6d>(NULL);
            return Vector6d::Zero();


        RandomLine3d tmpLine;
        vector<RandomPoint3d> rndpts3d;
        rndpts3d.reserve(pts3d.size());

        cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx,
                0, fy, cy,
                0, 0, 1);

        // compute uncertainty of 3d points
        for (auto &j : pts3d) {
            rndpts3d.push_back(compPt3dCov(j, K, 1));
        }
        // using ransac to extract a 3d line from 3d pts
        tmpLine = extract3dline_mahdist(rndpts3d);

        if (tmpLine.pts.size() / len > 0.4 && cv::norm(tmpLine.A - tmpLine.B) > 0.02) {
            //this line is reliable

            Vector6d line3D;
            line3D << tmpLine.A.x, tmpLine.A.y, tmpLine.A.z, tmpLine.B.x, tmpLine.B.y, tmpLine.B.z;

            cv::Mat Ac = (Mat_<float>(3, 1) << line3D(0), line3D(1), line3D(2));
            cv::Mat A = mRwc * Ac + mOw;
            cv::Mat Bc = (Mat_<float>(3, 1) << line3D(3), line3D(4), line3D(5));
            cv::Mat B = mRwc * Bc + mOw;
            line3D << A.at<float>(0, 0), A.at<float>(1, 0), A.at<float>(2, 0),
                    B.at<float>(0, 0), B.at<float>(1, 0), B.at<float>(2, 0);
            return line3D;
        } else {
            //return static_cast<Vector6d>(NULL);
            return Vector6d::Zero();
        }
    }


    
void    Frame::ExtractPlanes(const cv::Mat &imRGB, const cv::Mat &imDepth, const cv::Mat &K, const float &depthMapFactor) {
        planeDetector.readColorImage(imRGB);
        planeDetector.readDepthImage(mpTracker, imDepth, K, depthMapFactor);
        planeDetector.runPlaneDetection();

        for (int i = 0; i < planeDetector.plane_num_; i++) {
            auto &indices = planeDetector.plane_vertices_[i];
            PointCloud::Ptr inputCloud(new PointCloud());
            for (int j : indices) {
                PointT p;
                p.x = (float) planeDetector.cloud.vertices[j][0];
                p.y = (float) planeDetector.cloud.vertices[j][1];
                p.z = (float) planeDetector.cloud.vertices[j][2];
                p.r = static_cast<uint8_t>(planeDetector.cloud.verticesColour[j][0]);
                p.g = static_cast<uint8_t>(planeDetector.cloud.verticesColour[j][1]);
                p.b = static_cast<uint8_t>(planeDetector.cloud.verticesColour[j][2]);

                inputCloud->points.push_back(p);
            }

            auto extractedPlane = planeDetector.plane_filter.extractedPlanes[i];
            double nx = extractedPlane->normal[0];
            double ny = extractedPlane->normal[1];
            double nz = extractedPlane->normal[2];
            double cx = extractedPlane->center[0];
            double cy = extractedPlane->center[1];
            double cz = extractedPlane->center[2];

            float d = (float) -(nx * cx + ny * cy + nz * cz);

            pcl::VoxelGrid<PointT> voxel;
            //voxel.setLeafSize(0.2, 0.2, 0.2);
            voxel.setLeafSize(0.5, 0.5, 0.5);

            // 对点云进行分割处理
            // 假设我们将点云分割成 4 个块（你可以根据需要调整）
            float min_x = std::numeric_limits<float>::max();
            float min_y = std::numeric_limits<float>::max();
            float min_z = std::numeric_limits<float>::max();
            float max_x = -std::numeric_limits<float>::max();
            float max_y = -std::numeric_limits<float>::max();
            float max_z = -std::numeric_limits<float>::max();

            // 找到点云的边界
            for (const auto& point : inputCloud->points) {
                min_x = std::min(min_x, point.x);
                min_y = std::min(min_y, point.y);
                min_z = std::min(min_z, point.z);
                max_x = std::max(max_x, point.x);
                max_y = std::max(max_y, point.y);
                max_z = std::max(max_z, point.z);
            }

            // 设定分割块的尺寸（假设分割成 2x2x2 的网格）
            float block_size_x = (max_x - min_x) / 2;
            float block_size_y = (max_y - min_y) / 2;
            float block_size_z = (max_z - min_z) / 2;

            // 存储降采样后的结果
            PointCloud::Ptr coarseCloud(new PointCloud());

            for (float x = min_x; x < max_x; x += block_size_x) {
                for (float y = min_y; y < max_y; y += block_size_y) {
                    for (float z = min_z; z < max_z; z += block_size_z) {
                        // 使用 `pcl::CropBox` 来分割点云
                        pcl::CropBox<PointT> cropBoxFilter;
                        cropBoxFilter.setMin(Eigen::Vector4f(x, y, z, 1.0f));
                        cropBoxFilter.setMax(Eigen::Vector4f(x + block_size_x, y + block_size_y, z + block_size_z, 1.0f));
                        cropBoxFilter.setInputCloud(inputCloud);
                        
                        PointCloud::Ptr croppedCloud(new PointCloud());
                        cropBoxFilter.filter(*croppedCloud);

                        // 如果分割后的点云非空，则进行降采样
                        if (!croppedCloud->empty()) {
                            voxel.setInputCloud(croppedCloud);
                            PointCloud::Ptr blockCloud(new PointCloud());
                            voxel.filter(*blockCloud);

                            // 将降采样后的块加入到最终的点云中
                            *coarseCloud += *blockCloud;
                        }
                    }
                }
            }

            // 如果降采样后的点云为空，则跳过该平面
            if (coarseCloud->points.empty()) {
                continue;
            }


            // PointCloud::Ptr coarseCloud(new PointCloud());
            // voxel.setInputCloud(inputCloud);
            // voxel.filter(*coarseCloud);

            cv::Mat coef = (cv::Mat_<float>(4, 1) << nx, ny, nz, d);

            bool valid = MaxPointDistanceFromPlane(coef, coarseCloud);

            if (!valid) {
                continue;
            }

            mvPlanePoints.push_back(*coarseCloud);
            mvPlaneCoefficients.push_back(coef);
        }
    }

    cv::Mat Frame::ComputePlaneWorldCoeff(const int &idx) {
        cv::Mat temp;
        cv::transpose(mTcw, temp);
        return temp * mvPlaneCoefficients[idx];
    }

    bool Frame::MaxPointDistanceFromPlane(cv::Mat &plane, PointCloud::Ptr pointCloud) {
        bool erased = false;
        double threshold = 0.04;
        int i = 0;
        auto &points = pointCloud->points;

        for (auto &p : points) {
            double absDis = abs(plane.at<float>(0) * p.x +
                                plane.at<float>(1) * p.y +
                                plane.at<float>(2) * p.z +
                                plane.at<float>(3));

            if (absDis > mfDisTh)
                return false;

            i++;
        }

        if (points.size() < 4) {
            return false;
        }
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<PointT> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(mfDisTh);

        seg.setInputCloud(pointCloud);
        seg.segment(*inliers, *coefficients);

        float oldVal = plane.at<float>(3);
        float newVal = coefficients->values[3];

        cv::Mat oldPlane = plane.clone();

        plane.at<float>(0) = coefficients->values[0];
        plane.at<float>(1) = coefficients->values[1];
        plane.at<float>(2) = coefficients->values[2];
        plane.at<float>(3) = coefficients->values[3];

        if ((newVal < 0 && oldVal > 0) || (newVal > 0 && oldVal < 0)) {
            plane = -plane;
        }

        return true;
    }
} //namespace ORB_SLAM
