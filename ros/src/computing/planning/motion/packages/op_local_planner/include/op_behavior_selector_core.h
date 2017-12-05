/*
// *  Copyright (c) 2017, Nagoya University
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither the name of Autoware nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef OP_BEHAVIOR_SELECTOR_CORE
#define OP_BEHAVIOR_SELECTOR_CORE

// ROS includes
#include <ros/ros.h>
#include "PlannerCommonDef.h"
#include "TrajectoryCosts.h"
#include "DecisionMaker.h"

#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/Odometry.h>
#include <autoware_msgs/LaneArray.h>
#include <std_msgs/Int32.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <autoware_msgs/CanInfo.h>
#include <autoware_msgs/DetectedObjectArray.h>
#include <autoware_msgs/traffic_light.h>
#include <autoware_msgs/Signals.h>
#include <visualization_msgs/MarkerArray.h>


namespace BehaviorGeneratorNS
{

class BehaviorGen
{
protected: //Planning Related variables

	geometry_msgs::Pose m_OriginPos;
	PlannerHNS::WayPoint m_CurrentPos;
	bool bNewCurrentPos;

	PlannerHNS::VehicleState m_VehicleStatus;
	bool bVehicleStatus;

	std::vector<std::vector<PlannerHNS::WayPoint> > m_GlobalPaths;
	std::vector<PlannerHNS::WayPoint> t_centerTrajectorySmoothed;
	bool bWayGlobalPath;
	std::vector<std::vector<PlannerHNS::WayPoint> > m_RollOuts;
	bool bRollOuts;

	PlannerHNS::MAP_SOURCE_TYPE m_MapType;
	std::string m_MapPath;

	PlannerHNS::RoadNetwork m_Map;
	bool bMap;

	std::vector<PlannerHNS::TrajectoryCost> m_TrajectoryCosts;
	PlannerHNS::TrajectoryCost m_TrajectoryBestCost;
	bool bBestCost;

	PlannerHNS::DecisionMaker m_BehaviorGenerator;
	PlannerHNS::BehaviorState m_CurrentBehavior;

  	std::vector<std::string>    m_LogData;

  	PlannerHNS::PlanningParams m_PlanningParams;
  	PlannerHNS::CAR_BASIC_INFO m_CarInfo;

  	autoware_msgs::lane m_CurrentTrajectoryToSend;
  	bool bNewLightStatus;
	bool bNewLightSignal;
	PlannerHNS::TrafficLightState  m_CurrLightStatus;
	std::vector<PlannerHNS::TrafficLight> m_CurrTrafficLight;
	std::vector<PlannerHNS::TrafficLight> m_PrevTrafficLight;

protected: //ROS messages (topics)
	ros::NodeHandle nh;

	//define publishers
	ros::Publisher pub_LocalPath;
	ros::Publisher pub_LocalBasePath;
	ros::Publisher pub_ClosestIndex;
	ros::Publisher pub_BehaviorState;
	ros::Publisher pub_SimuBoxPose;

	// define subscribers.
	ros::Subscriber sub_current_pose 		;
	ros::Subscriber sub_current_velocity	;
	ros::Subscriber sub_robot_odom			;
	ros::Subscriber sub_can_info			;
	ros::Subscriber sub_GlobalPlannerPaths	;
	ros::Subscriber sub_LocalPlannerPaths	;
	ros::Subscriber sub_TrafficLightStatus  ;
	ros::Subscriber sub_TrafficLightSignals ;
	ros::Subscriber sub_Trajectory_Cost	    ;
	ros::Publisher pub_BehaviorStateRviz;

protected: // Callback function for subscriber.
	void callbackGetCurrentPose(const geometry_msgs::PoseStampedConstPtr& msg);
	void callbackGetVehicleStatus(const geometry_msgs::TwistStampedConstPtr& msg);
	void callbackGetCanInfo(const autoware_msgs::CanInfoConstPtr &msg);
	void callbackGetRobotOdom(const nav_msgs::OdometryConstPtr& msg);
	void callbackGetGlobalPlannerPath(const autoware_msgs::LaneArrayConstPtr& msg);
	void callbackGetLocalPlannerPath(const autoware_msgs::LaneArrayConstPtr& msg);
	void callbackGetLocalTrajectoryCost(const autoware_msgs::laneConstPtr& msg);
	void callbackGetTrafficLightStatus(const autoware_msgs::traffic_light & msg);
	void callbackGetTrafficLightSignals(const autoware_msgs::Signals& msg);

protected: //Helper Functions
  void UpdatePlanningParams(ros::NodeHandle& _nh);
  void SendLocalPlanningTopics();
  void VisualizeLocalPlanner();

public:
  BehaviorGen();
  ~BehaviorGen();
  void MainLoop();
};

}

#endif  // OP_BEHAVIOR_SELECTOR_CORE
