import re
import sys
import os
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple
 
 
# ─────────────────────────────────────────────────────────────
# COMPREHENSIVE PATTERN LIBRARY — ALL ROAD SCENARIOS
# Total: 20 categories, 150+ patterns
# ─────────────────────────────────────────────────────────────
 
LOW_CONFIDENCE_PATTERNS = [
 
    # ══════════════════════════════════════════════════════════
    # CATEGORY 1 — VISUAL UNCERTAINTY
    # Covers: what the vehicle/driver sees on road
    # ══════════════════════════════════════════════════════════
    {
        "category": "Visual Uncertainty",
        "patterns": [
            (r"\blooks?\s+like\b",        "State what it is factually",              "is"),
            (r"\bappears to be\b",        "State what it is — not a guess",          "is"),
            (r"\bappears\b",              "Use factual observation",                 "is"),
            (r"\bseems to be\b",          "Avoid assumption — state the fact",       "is"),
            (r"\bseems\b",                "Avoid assumption language",               "is"),
            (r"\bI think\b",              "Remove subjective language",              ""),
            (r"\bI believe\b",            "Remove subjective language",              ""),
            (r"\bI feel\b",               "Remove subjective language",              ""),
            (r"\bpossibly\b",             "Use certain language",                    ""),
            (r"\bprobably\b",             "Use certain language",                    ""),
            (r"\bperhaps\b",              "Use certain language",                    ""),
            (r"\bmaybe\b",                "Use certain language",                    ""),
            (r"\bit seems\b",             "State the observation directly",          ""),
            (r"\bit looks\b",             "State the observation directly",          ""),
            (r"\bI notice\b",             "State the observation directly",          ""),
            (r"\bI can see\b",            "Remove subjective — state the object",    ""),
            (r"\bI observe\b",            "State the observation directly",          ""),
        ]
    },
 
    # ══════════════════════════════════════════════════════════
    # CATEGORY 2 — MODAL UNCERTAINTY
    # Covers: uncertain predictions about vehicle behavior
    # ══════════════════════════════════════════════════════════
    {
        "category": "Modal Uncertainty",
        "patterns": [
            (r"\bmight be\b",             "Use definitive statement",                "is"),
            (r"\bmight\b",                "Use definitive statement",                "will"),
            (r"\bcould be\b",             "Use definitive statement",                "is"),
            (r"\bcould\b",                "Use definitive statement",                "will"),
            (r"\bwould be\b",             "Use definitive statement",                "is"),
            (r"\bshould be\b",            "Use factual observation",                 "is"),
            (r"\bmay be\b",               "Avoid hedging language",                  "is"),
            (r"\bmay\b",                  "Avoid hedging language",                  "will"),
            (r"\bought to\b",             "Use direct statement",                    "will"),
            (r"\bwould likely\b",         "Remove hedging — use direct statement",   "will"),
        ]
    },
 
    # ══════════════════════════════════════════════════════════
    # CATEGORY 3 — WEAK / HEDGED ACTIONS
    # Covers: ADV/vehicle action descriptions
    # ══════════════════════════════════════════════════════════
    {
        "category": "Weak Actions",
        "patterns": [
            (r"\btrying to\b",            "State the action directly",               ""),
            (r"\battempting to\b",        "State the action directly",               ""),
            (r"\bexpected to\b",          "Use factual description",                 "will"),
            (r"\blikely to\b",            "Use factual description",                 "will"),
            (r"\blikely\b",               "Remove hedging adverb",                   ""),
            (r"\bintend(s|ing)? to\b",    "Use direct action statement",             "will"),
            (r"\bplan(s|ning)? to\b",     "Use direct action statement",             "will"),
            (r"\bhoping to\b",            "Use direct action statement",             "will"),
            (r"\baiming to\b",            "Use direct action statement",             "will"),
            (r"\bstarting to\b",          "Use direct action — state it factually",  ""),
            (r"\bbeginning to\b",         "Use direct action — state it factually",  ""),
            (r"\babout to\b",             "State the action directly",               "will"),
            (r"\bgoing to\b",             "Use 'will' for direct intention",         "will"),
            (r"\bseems to be (moving|stopping|turning|slowing|accelerating)\b",
                                          "State the action directly",               "is"),
        ]
    },
 
    # ══════════════════════════════════════════════════════════
    # CATEGORY 4 — ACTION PRECISION
    # Covers: imprecise descriptions of vehicle movements
    # ══════════════════════════════════════════════════════════
    {
        "category": "Action Precision",
        "patterns": [
            (r"\btaking a right turn\b",  "Use 'turning right'",                    "turning right"),
            (r"\btaking a left turn\b",   "Use 'turning left'",                     "turning left"),
            (r"\btaking a turn\b",        "Specify direction: turning right/left",   "turning"),
            (r"\bmaking a right turn\b",  "Use 'turning right'",                    "turning right"),
            (r"\bmaking a left turn\b",   "Use 'turning left'",                     "turning left"),
            (r"\bmaking a U-turn\b",      "Use 'performing a U-turn'",              "performing a U-turn"),
            (r"\bgoing straight\b",       "Use 'continuing straight'",              "continuing straight"),
            (r"\bslowing down\b",         "Use 'decelerating'",                     "decelerating"),
            (r"\bspeeding up\b",          "Use 'accelerating'",                     "accelerating"),
            (r"\bpicking up speed\b",     "Use 'accelerating'",                     "accelerating"),
            (r"\blosing speed\b",         "Use 'decelerating'",                     "decelerating"),
            (r"\bcoming to a stop\b",     "Use 'stopping' or state final position", "stopping"),
            (r"\bpulling over\b",         "Use 'moving to the right side and stopping'", "moving to the right side"),
            (r"\bpulling out\b",          "Use 'merging into traffic'",             "merging into traffic"),
            (r"\bpulling in\b",           "Use 'entering the lane'",               "entering the lane"),
            (r"\bcutting in\b",           "Use 'merging abruptly into the lane'",  "merging abruptly into the lane"),
            (r"\bchanging lanes?\b",      "Specify direction: merging left/right",  "merging"),
            (r"\bswitching lanes?\b",     "Specify direction: merging left/right",  "merging"),
            (r"\bmoving over\b",          "Specify direction of lane change",       ""),
            (r"\bdrifting (left|right)\b","Use 'moving left/right'",               "moving"),
            (r"\bveering (left|right)\b", "Use 'moving left/right'",               "moving"),
            (r"\bswerving\b",             "Use 'making an abrupt lateral movement'","making an abrupt lateral movement"),
            (r"\bjerking\b",              "Use 'making a sudden movement'",         "making a sudden movement"),
            (r"\breverting\b",            "Use 'returning to previous path'",       "returning to previous path"),
            (r"\bbacking up\b",           "Use 'reversing'",                        "reversing"),
            (r"\bgoing in reverse\b",     "Use 'reversing'",                        "reversing"),
        ]
    },
 
    # ══════════════════════════════════════════════════════════
    # CATEGORY 5 — SPATIAL PRECISION
    # Covers: imprecise location/direction descriptions
    # ══════════════════════════════════════════════════════════
    {
        "category": "Spatial Precision",
        "patterns": [
            (r"\bin the front\b",         "Use 'ahead'",                            "ahead"),
            (r"\bin front of me\b",       "Use 'ahead'",                            "ahead"),
            (r"\bup ahead\b",             "Use 'ahead'",                            "ahead"),
            (r"\bover there\b",           "Specify exact direction/distance",        ""),
            (r"\bnearby\b",               "Specify exact distance in meters",        ""),
            (r"\bsomewhere\b",            "Specify exact location",                  ""),
            (r"\bfar away\b",             "Use exact distance in meters",           ""),
            (r"\bclose by\b",             "Use exact distance in meters",           ""),
            (r"\bjust ahead\b",           "Specify exact distance in meters",        "ahead"),
            (r"\bright next to\b",        "Specify lane position or distance",       "adjacent to"),
            (r"\bvery close\b",           "Use exact distance in meters",           ""),
            (r"\bvery far\b",             "Use exact distance in meters",           ""),
            (r"\ba bit\b",                "Use precise measurement",                 ""),
            (r"\ba little\b",             "Use precise measurement",                 ""),
            (r"\bquite far\b",            "Use exact distance in meters",           ""),
            (r"\bnot far\b",              "Use exact distance in meters",           ""),
            (r"\bin the distance\b",      "Use exact distance or 'beyond X meters'", ""),
            (r"\baround the corner\b",    "Use 'at the upcoming intersection'",     "at the upcoming intersection"),
            (r"\bjust behind\b",          "Specify exact distance in meters",        "behind"),
            (r"\bright behind\b",         "Specify exact distance in meters",        "directly behind"),
            (r"\bdirectly in front\b",    "Use 'ahead in the same lane'",           "ahead in the same lane"),
            (r"\bsome distance\b",        "Use exact distance in meters",           ""),
        ]
    },
 
    # ══════════════════════════════════════════════════════════
    # CATEGORY 6 — VEHICLE DESCRIPTION IMPRECISION
    # Covers: vague vehicle type/color/state descriptions
    # ══════════════════════════════════════════════════════════
    {
        "category": "Vehicle Description",
        "patterns": [
            (r"\bsome (car|vehicle|truck|van)\b",
                                          "Name the vehicle type specifically",      "a"),
            (r"\ba (car|vehicle)\b",      "Specify vehicle type: SUV, sedan, truck", ""),
            (r"\banother (car|vehicle)\b","Specify vehicle type specifically",        ""),
            (r"\bone of the (cars|vehicles)\b",
                                          "Reference the specific vehicle",          "the"),
            (r"\bthe (car|vehicle) in question\b",
                                          "Name the specific vehicle",               "the"),
            (r"\bsomething on the road\b","Identify the object specifically",        "an object"),
            (r"\ban object\b",            "Identify the specific object type",       ""),
            (r"\bsome kind of\b",         "Identify the vehicle/object specifically",""),
            (r"\bwhat appears to be\b",   "State what it is directly",              ""),
            (r"\bwhat looks like\b",      "State what it is directly",              ""),
            (r"\ba (big|large|huge) (car|vehicle|truck)\b",
                                          "Use specific vehicle type not size adj",  "a"),
            (r"\bmoving (car|vehicle|object)\b",
                                          "Specify the type and direction of movement",""),
            (r"\bparked (car|vehicle)\b", "Specify vehicle type and exact position", ""),
            (r"\bstalled (car|vehicle)\b","Use 'a stationary vehicle'",             "a stationary vehicle"),
        ]
    },
 
    # ══════════════════════════════════════════════════════════
    # CATEGORY 7 — SPEED DESCRIPTION IMPRECISION
    # Covers: imprecise speed-related language
    # ══════════════════════════════════════════════════════════
    {
        "category": "Speed Description",
        "patterns": [
            (r"\bvery (fast|quickly|rapidly|slow|slowly)\b",
                                          "Use exact speed in m/s or mph",          ""),
            (r"\bfast(ly)?\b",            "Use exact speed value in m/s",           ""),
            (r"\bslowly\b",               "Use exact speed value in m/s",           ""),
            (r"\bquickly\b",              "Use exact speed value in m/s",           ""),
            (r"\brapidly\b",              "Use exact speed value in m/s",           ""),
            (r"\bat (a )?(high|low|moderate|normal) speed\b",
                                          "State exact speed in m/s",               ""),
            (r"\bmoving (fast|slow|quickly|rapidly)\b",
                                          "Use exact speed in m/s",                 "moving"),
            (r"\bcrawling\b",             "Use exact speed in m/s",                 "moving slowly"),
            (r"\bzooming\b",              "Use exact speed in m/s",                 "moving"),
            (r"\bbarreling\b",            "Use exact speed in m/s",                 "moving"),
            (r"\bat speed\b",             "State exact speed value",                 ""),
            (r"\bfull speed\b",           "State exact speed value in m/s",         ""),
            (r"\bat (a )?low speed\b",    "State exact speed value in m/s",         ""),
            (r"\bnormal speed\b",         "State exact speed in m/s",               ""),
            (r"\bregular speed\b",        "State exact speed in m/s",               ""),
        ]
    },
 
    # ══════════════════════════════════════════════════════════
    # CATEGORY 8 — TRAFFIC SIGNAL & SIGN IMPRECISION
    # Covers: traffic lights, signs, signals
    # ══════════════════════════════════════════════════════════
    {
        "category": "Traffic Signal Description",
        "patterns": [
            (r"\bthe light\b",            "Specify 'traffic signal' and its state (red/green/yellow)", "the traffic signal"),
            (r"\bred light\b",            "Use 'red traffic signal'",               "red traffic signal"),
            (r"\bgreen light\b",          "Use 'green traffic signal'",             "green traffic signal"),
            (r"\byellow light\b",         "Use 'yellow traffic signal'",            "yellow traffic signal"),
            (r"\bchanging light\b",       "Specify signal transition: red to green", "changing traffic signal"),
            (r"\btraffic light\b",        "Use 'traffic signal'",                   "traffic signal"),
            (r"\bthe sign\b",             "Specify sign type: stop sign, yield sign","the road sign"),
            (r"\bthe stop sign\b",        "Confirm 'stop sign is present at X'",    "the stop sign"),
            (r"\bthe signal\b",           "Specify signal type and current state",   "the traffic signal"),
            (r"\bflashing (light|signal)\b",
                                          "Specify: flashing red or flashing yellow","flashing signal"),
            (r"\bblinking\b",             "Specify signal type and color",           "flashing"),
        ]
    },
 
    # ══════════════════════════════════════════════════════════
    # CATEGORY 9 — PEDESTRIAN & CYCLIST IMPRECISION
    # Covers: people and cyclists on or near road
    # ══════════════════════════════════════════════════════════
    {
        "category": "Pedestrian and Cyclist",
        "patterns": [
            (r"\bsomeone\b",              "Specify: pedestrian, cyclist, or person", "a pedestrian"),
            (r"\ba person\b",             "Specify role: pedestrian or cyclist",     "a pedestrian"),
            (r"\bpeople\b",               "Specify: pedestrians, cyclists",          "pedestrians"),
            (r"\bwalking (person|individual)\b",
                                          "Use 'pedestrian'",                        "pedestrian"),
            (r"\ba (guy|man|woman|child|kid)\b",
                                          "Use 'a pedestrian' in AV context",        "a pedestrian"),
            (r"\bthe (guy|man|woman|child|kid)\b",
                                          "Use 'the pedestrian'",                    "the pedestrian"),
            (r"\bcrossing the (road|street|lane)\b",
                                          "Confirm if in crosswalk or not",          "crossing"),
            (r"\bwalking into (the road|traffic)\b",
                                          "Use 'entering the roadway'",             "entering the roadway"),
            (r"\bbiker\b",                "Use 'cyclist'",                           "cyclist"),
            (r"\bbike\b",                 "Use 'bicycle' or 'cyclist'",             "bicycle"),
            (r"\bmotorbike\b",            "Use 'motorcycle'",                        "motorcycle"),
            (r"\bscooter\b",              "Use 'scooter (two-wheeled vehicle)'",    "scooter"),
            (r"\bjogging\b",              "Use 'running' or 'moving on foot'",       "running"),
            (r"\bhanging around\b",       "Use 'standing near the roadway'",        "standing near the roadway"),
        ]
    },
 
    # ══════════════════════════════════════════════════════════
    # CATEGORY 10 — ROAD CONDITION IMPRECISION
    # Covers: road surface, lane, markings, hazards
    # ══════════════════════════════════════════════════════════
    {
        "category": "Road Condition",
        "patterns": [
            (r"\bbad road\b",             "Specify: potholes, cracks, uneven surface","damaged road surface"),
            (r"\brough road\b",           "Specify: potholes, gravel, uneven",       "uneven road surface"),
            (r"\bslippery\b",             "Specify: wet, icy, or oily road surface", "low-traction surface"),
            (r"\bwet road\b",             "Use 'wet road surface'",                  "wet road surface"),
            (r"\bicy road\b",             "Use 'ice-covered road surface'",          "ice-covered road surface"),
            (r"\bdirty road\b",           "Specify: gravel, debris, mud",            "road with debris"),
            (r"\bthe road is (bad|good|okay)\b",
                                          "Specify exact road condition",            "the road surface"),
            (r"\bsome debris\b",          "Specify type: rock, branch, object",      "road debris"),
            (r"\ban obstacle\b",          "Identify the specific obstacle",          ""),
            (r"\bsomething in the road\b","Identify the object specifically",        "an object in the road"),
            (r"\bthe road curves\b",      "Specify curve direction: curves left/right","the road curves"),
            (r"\bundulating road\b",      "Use 'uneven road surface'",              "uneven road surface"),
            (r"\bpothole(s)?\b",          "Confirm exact location of pothole",       "pothole"),
            (r"\bspeed bump\b",           "Confirm 'speed bump is present at X m'", "speed bump"),
            (r"\brough patch\b",          "Use 'uneven road surface section'",       "uneven road section"),
        ]
    },
 
    # ══════════════════════════════════════════════════════════
    # CATEGORY 11 — WEATHER & VISIBILITY IMPRECISION
    # Covers: weather conditions affecting driving
    # ══════════════════════════════════════════════════════════
    {
        "category": "Weather and Visibility",
        "patterns": [
            (r"\bweather is (bad|good|okay|fine)\b",
                                          "Specify: rain, fog, sun glare, clear",   "weather condition"),
            (r"\bpoor visibility\b",      "Specify cause: fog, rain, night, glare",  "reduced visibility"),
            (r"\bhard to see\b",          "Specify cause: fog, glare, darkness",     ""),
            (r"\bbright (sun|light|glare)\b",
                                          "Use 'sun glare reducing visibility'",    "sun glare"),
            (r"\ba bit foggy\b",          "Use 'fog reducing visibility'",           "fog"),
            (r"\bslightly (rainy|wet|foggy)\b",
                                          "State exact weather condition",           ""),
            (r"\bvery (sunny|bright|dark|cloudy)\b",
                                          "State exact condition and its effect",    ""),
            (r"\bI can('t| not) see (well|clearly|far)\b",
                                          "Specify cause of reduced visibility",     ""),
            (r"\bdark outside\b",         "Use 'low ambient light conditions'",      "low ambient light"),
            (r"\bright time\b",           "Use 'daytime conditions'",               "daytime"),
            (r"\bnight time\b",           "Use 'nighttime conditions'",             "nighttime"),
            (r"\bwindy\b",                "Specify wind effect on visibility/control","wind present"),
        ]
    },
 
    # ══════════════════════════════════════════════════════════
    # CATEGORY 12 — INTERSECTION & JUNCTION IMPRECISION
    # Covers: intersections, junctions, crossings
    # ══════════════════════════════════════════════════════════
    {
        "category": "Intersection Description",
        "patterns": [
            (r"\bthe junction\b",         "Use 'the intersection at [location]'",   "the intersection"),
            (r"\bthe crossing\b",         "Specify: crosswalk, intersection, or railroad crossing", "the intersection"),
            (r"\bthe corner\b",           "Use 'the intersection ahead'",           "the intersection"),
            (r"\bT-junction\b",           "Use 'T-shaped intersection'",            "T-shaped intersection"),
            (r"\bthe turn\b",             "Specify: right turn, left turn, U-turn", ""),
            (r"\bthe upcoming (turn|corner|junction)\b",
                                          "Specify exact type and direction",        "the upcoming intersection"),
            (r"\ba four-way\b",           "Use 'a four-way intersection'",          "a four-way intersection"),
            (r"\bthe roundabout\b",       "Use 'the roundabout ahead'",             "the roundabout ahead"),
            (r"\bthe merge\b",            "Use 'the merge point ahead'",            "the merge point ahead"),
            (r"\bthe exit\b",             "Specify: highway exit or ramp",          "the exit ramp"),
        ]
    },
 
    # ══════════════════════════════════════════════════════════
    # CATEGORY 13 — LANE DESCRIPTION IMPRECISION
    # Covers: lanes, lane positions, lane markings
    # ══════════════════════════════════════════════════════════
    {
        "category": "Lane Description",
        "patterns": [
            (r"\bmy lane\b",              "Use 'the current lane' or 'the ego lane'","the current lane"),
            (r"\bthe other lane\b",       "Specify: left lane, right lane, adjacent lane", "the adjacent lane"),
            (r"\bthe wrong lane\b",       "Use 'the oncoming traffic lane'",        "the oncoming traffic lane"),
            (r"\bthe fast lane\b",        "Use 'the leftmost/passing lane'",        "the passing lane"),
            (r"\bthe slow lane\b",        "Use 'the rightmost lane'",               "the rightmost lane"),
            (r"\bthe middle lane\b",      "Use 'the center lane'",                  "the center lane"),
            (r"\bthe bike lane\b",        "Use 'the designated bicycle lane'",      "the designated bicycle lane"),
            (r"\bthe turn lane\b",        "Specify: right-turn lane or left-turn lane", "the turn lane"),
            (r"\bblocking (my|the) lane\b",
                                          "Use 'obstructing the current lane'",     "obstructing the current lane"),
            (r"\bstraddling (the )?lanes?\b",
                                          "Use 'positioned across two lanes'",      "positioned across two lanes"),
            (r"\bout of (my|the) lane\b", "Use 'crossing the lane boundary'",       "crossing the lane boundary"),
        ]
    },
 
    # ══════════════════════════════════════════════════════════
    # CATEGORY 14 — PARKING & STATIONARY VEHICLE IMPRECISION
    # Covers: parked, stopped, standing vehicles
    # ══════════════════════════════════════════════════════════
    {
        "category": "Parking and Stationary",
        "patterns": [
            (r"\bparked (somewhere|there|here)\b",
                                          "Specify exact parking position",          "parked"),
            (r"\bstopped (somewhere|there)\b",
                                          "Specify location and reason if known",    "stopped"),
            (r"\bstanding still\b",       "Use 'stationary'",                       "stationary"),
            (r"\bnot moving\b",           "Use 'stationary'",                       "stationary"),
            (r"\bdouble parked\b",        "Use 'parked in the travel lane'",        "parked in the travel lane"),
            (r"\billegal(ly)? parked\b",  "Use 'parked in a restricted zone'",      "parked in a restricted zone"),
            (r"\bblocking (the road|traffic|the lane)\b",
                                          "Use 'obstructing the lane'",             "obstructing the lane"),
            (r"\bpulled to the side\b",   "Use 'stopped on the right shoulder'",    "stopped on the right shoulder"),
            (r"\bon the side of the road\b",
                                          "Use 'on the road shoulder'",             "on the road shoulder"),
            (r"\bhalfway in the road\b",  "Use 'partially blocking the lane'",      "partially blocking the lane"),
        ]
    },
 
    # ══════════════════════════════════════════════════════════
    # CATEGORY 15 — EMERGENCY VEHICLE IMPRECISION
    # Covers: ambulance, police, fire truck scenarios
    # ══════════════════════════════════════════════════════════
    {
        "category": "Emergency Vehicle",
        "patterns": [
            (r"\bthe (ambulance|fire truck|police car) is coming\b",
                                          "Specify: approaching from which direction",""),
            (r"\bemergency vehicle\b",    "Specify type: ambulance, fire truck, police", ""),
            (r"\bwith (its )?siren(s)?\b","Specify: audible siren and/or flashing lights", "with active siren"),
            (r"\bflashing (red and blue|blue and red) lights\b",
                                          "Use 'active emergency lighting'",        "active emergency lighting"),
            (r"\bpolice car\b",           "Confirm if marked or unmarked patrol vehicle", "marked patrol vehicle"),
            (r"\bfire truck\b",           "Confirm vehicle type as fire apparatus",  "fire apparatus"),
            (r"\brush(ing)? past\b",      "Use 'passing at high speed with siren'", "passing at high speed"),
            (r"\bpull(ing)? over (for|to allow)\b",
                                          "Use 'yielding to emergency vehicle'",    "yielding to emergency vehicle"),
        ]
    },
 
    # ══════════════════════════════════════════════════════════
    # CATEGORY 16 — CONSTRUCTION ZONE IMPRECISION
    # Covers: construction zones, road works, barriers
    # ══════════════════════════════════════════════════════════
    {
        "category": "Construction Zone",
        "patterns": [
            (r"\bconstruction (ahead|zone|site)\b",
                                          "Specify: lane closure, detour, or merge", "construction zone"),
            (r"\broad works?\b",          "Use 'active construction zone'",          "active construction zone"),
            (r"\bthe (cone|cones)\b",     "Use 'traffic cone(s) marking X'",        "traffic cone"),
            (r"\bbarrier(s)?\b",          "Specify type: jersey barrier, water barrier, cone", "road barrier"),
            (r"\bdetour\b",               "Use 'mandatory detour route'",            "mandatory detour"),
            (r"\blane closed\b",          "Use 'lane closure ahead'",               "lane closure ahead"),
            (r"\bmerge (left|right|ahead)\b",
                                          "Use 'mandatory lane merge'",             "mandatory lane merge"),
            (r"\bwork(ers)? (on|in) the road\b",
                                          "Use 'construction workers present in roadway'","construction workers in roadway"),
            (r"\bflagger\b",              "Use 'traffic control flagger'",           "traffic control flagger"),
        ]
    },
 
    # ══════════════════════════════════════════════════════════
    # CATEGORY 17 — ANIMAL ON ROAD IMPRECISION
    # Covers: animals crossing or on road
    # ══════════════════════════════════════════════════════════
    {
        "category": "Animal on Road",
        "patterns": [
            (r"\ban animal\b",            "Specify animal type: dog, deer, bird etc",""),
            (r"\bsome animal\b",          "Specify animal type",                     "an animal"),
            (r"\ba (dog|cat|deer|bird|rabbit|squirrel) (crossing|in|on) (the road|the lane)\b",
                                          "Confirm exact position in lane",          ""),
            (r"\bwildlife\b",             "Specify animal type on road",             "an animal"),
            (r"\bthe animal ran\b",       "Specify: animal crossed lane, exited scene",""),
        ]
    },
 
    # ══════════════════════════════════════════════════════════
    # CATEGORY 18 — APPROXIMATION & VAGUE QUANTITIES
    # Covers: imprecise numbers and measurements
    # ══════════════════════════════════════════════════════════
    {
        "category": "Approximation",
        "patterns": [
            (r"\babout\b",                "Use exact value if known",               "approximately"),
            (r"\broughly\b",              "Use exact value if known",               "approximately"),
            (r"\bsomething like\b",       "Be specific",                             ""),
            (r"\bkind of\b",              "Remove filler phrase",                    ""),
            (r"\bsort of\b",              "Remove filler phrase",                    ""),
            (r"\bsomewhat\b",             "Use precise description",                 ""),
            (r"\baround \d+\b",           "Use exact number if known",               ""),
            (r"\ba few (meters|seconds|cars)\b",
                                          "Use exact count or measurement",          ""),
            (r"\bseveral (meters|seconds|vehicles)\b",
                                          "Use exact count or measurement",          ""),
            (r"\bmany (cars|vehicles|pedestrians)\b",
                                          "Use exact count if known",                "multiple"),
            (r"\ba couple (of )?(cars|meters|seconds)\b",
                                          "Use exact number",                        "two"),
            (r"\ba lot of\b",             "Use exact count",                         "multiple"),
            (r"\ba number of\b",          "Use exact count",                         "multiple"),
        ]
    },
 
    # ══════════════════════════════════════════════════════════
    # CATEGORY 19 — VISIBILITY DOUBT
    # Covers: uncertain or partial observations
    # ══════════════════════════════════════════════════════════
    {
        "category": "Visibility Doubt",
        "patterns": [
            (r"\bnot\s+sure\b",           "Remove uncertainty expression",           ""),
            (r"\buncertain\b",            "Remove uncertainty expression",           ""),
            (r"\bhard to (tell|say|see)\b","Remove if object not confirmable",       ""),
            (r"\bI cannot (tell|say|confirm)\b",
                                          "Remove unverifiable claims",              ""),
            (r"\bI('m| am) not sure\b",   "Remove subjective uncertainty",           ""),
            (r"\bnot (entirely|fully|completely) (clear|visible|sure)\b",
                                          "Clarify or remove if unseen",             ""),
            (r"\bhard to see\b",          "Remove or specify cause",                 ""),
            (r"\bdifficult to (see|determine|tell)\b",
                                          "Remove unverifiable claims",              ""),
            (r"\bnot clearly visible\b",  "Omit if not clearly visible",            ""),
            (r"\bI can'?t (see|tell|confirm)\b",
                                          "Remove unverifiable claims",              ""),
            (r"\bpartially visible\b",    "Describe only what IS visible",           ""),
            (r"\bbarely visible\b",       "Remove or specify exact visibility condition",""),
            (r"\bnot fully visible\b",    "Remove or describe only visible portion", ""),
            (r"\bout of sight\b",         "Remove — object not in scene",            ""),
            (r"\bcan('t| not) make out\b","Remove unverifiable observation",         ""),
        ]
    },
 
    # ══════════════════════════════════════════════════════════
    # CATEGORY 20 — GRAMMAR FIX
    # Covers: contractions, informal grammar in captions
    # ══════════════════════════════════════════════════════════
    {
        "category": "Grammar Fix",
        "patterns": [
            (r"\bIt'?s\s+look\b",         "Should be 'It looks' or remove entirely", "The"),
            (r"\bIt'?s\s+looks\b",        "Should be 'It looks' — remove 'It's'",   "It"),
            (r"\bIt'?s\b",                "Avoid 'It's' — use subject noun instead", "The"),
            (r"\bThere'?s\b",             "Avoid 'There's' — be specific",           "A"),
            (r"\bthey'?re\b",             "Avoid contraction — use 'they are'",      "they are"),
            (r"\bwe'?re\b",               "Avoid contraction — use 'we are'",        "we are"),
            (r"\bdon'?t\b",               "Avoid contraction — use 'do not'",        "do not"),
            (r"\bcan'?t\b",               "Avoid contraction — use 'cannot'",        "cannot"),
            (r"\bwon'?t\b",               "Avoid contraction — use 'will not'",      "will not"),
            (r"\bisn'?t\b",               "Avoid contraction — use 'is not'",        "is not"),
            (r"\baren'?t\b",              "Avoid contraction — use 'are not'",       "are not"),
            (r"\bwasn'?t\b",              "Avoid contraction — use 'was not'",       "was not"),
            (r"\bweren'?t\b",             "Avoid contraction — use 'were not'",      "were not"),
            (r"\bdidn'?t\b",              "Avoid contraction — use 'did not'",       "did not"),
            (r"\bhasn'?t\b",              "Avoid contraction — use 'has not'",       "has not"),
            (r"\bhaven'?t\b",             "Avoid contraction — use 'have not'",      "have not"),
            (r"\bI'?m\b",                 "Avoid 'I'm' — rephrase sentence",         ""),
            (r"\bI'?ve\b",                "Avoid 'I've' — rephrase sentence",        ""),
            (r"\bI'?ll\b",                "Avoid 'I'll' — use 'I will'",             "I will"),
            (r"\bI'?d\b",                 "Avoid 'I'd' — rephrase sentence",         ""),
            (r"\byou'?re\b",              "Avoid contraction — use 'you are'",       "you are"),
            (r"\blet'?s\b",               "Avoid contraction — use 'let us'",        "let us"),
        ]
    },
]
 
 
# ─────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────
 
@dataclass
class DetectedIssue:
    word_found:  str
    category:    str
    reason:      str
    suggestion:  str
    position:    int
    sentence:    str
 
 
@dataclass
class DetectionResult:
    original_text:  str
    source_file:    str = "manual input"
    issues:         List[DetectedIssue] = field(default_factory=list)
    cleaned_text:   str = ""
    total_found:    int = 0
    categories_hit: List[str] = field(default_factory=list)
    passed:         bool = False
 
    def summary(self) -> str:
        if self.passed:
            return "✅ PASSED — No low confidence words found"
        return (
            f"❌ FAILED — {self.total_found} issue(s) found "
            f"in {len(self.categories_hit)} category(ies): "
            f"{', '.join(self.categories_hit)}"
        )
 
 
# ─────────────────────────────────────────────────────────────
# FILE INPUT HANDLER
# ─────────────────────────────────────────────────────────────
 
class FileInputHandler:
 
    @staticmethod
    def read_pdf(filepath: str) -> str:
        try:
            import pypdf
        except ImportError:
            print("❌ pypdf not installed. Run: pip install pypdf")
            sys.exit(1)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PDF not found: {filepath}")
        print(f"\n📄 Reading PDF: {filepath}")
        pages = []
        with open(filepath, "rb") as f:
            reader = pypdf.PdfReader(f)
            print(f"   Pages: {len(reader.pages)}")
            for i, page in enumerate(reader.pages):
                t = page.extract_text()
                if t and t.strip():
                    pages.append(t.strip())
                    print(f"   Page {i+1}: {len(t)} chars")
                else:
                    print(f"   Page {i+1}: ⚠️  no text (scanned?)")
        if not pages:
            raise ValueError("No text extracted — PDF may be scanned image.")
        full = "\n\n".join(pages)
        print(f"✅ PDF read: {len(full)} total chars")
        return full
 
    @staticmethod
    def read_txt(filepath: str) -> str:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        print(f"\n📝 Reading: {filepath}")
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                with open(filepath, "r", encoding=enc) as f:
                    text = f.read()
                print(f"✅ Read ({enc}) — {len(text)} chars")
                return text
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Cannot read file: {filepath}")
 
    @staticmethod
    def read_manual() -> str:
        print("\n✏️  Enter text (press ENTER twice when done):\n")
        lines, empty = [], 0
        while True:
            try:
                line = input()
                if line == "":
                    empty += 1
                    if empty >= 2:
                        break
                else:
                    empty = 0
                    lines.append(line)
            except EOFError:
                break
        text = "\n".join(lines).strip()
        if not text:
            raise ValueError("No text entered.")
        print(f"\n✅ Input received — {len(text)} chars")
        return text
 
 
# ─────────────────────────────────────────────────────────────
# DETECTOR CLASS
# ─────────────────────────────────────────────────────────────
 
class LowConfidenceDetector:
 
    def __init__(self):
        self._compiled = []
        for cat_block in LOW_CONFIDENCE_PATTERNS:
            cat = cat_block["category"]
            for (pattern, reason, suggestion) in cat_block["patterns"]:
                self._compiled.append({
                    "regex":      re.compile(pattern, re.IGNORECASE),
                    "pattern":    pattern,
                    "category":   cat,
                    "reason":     reason,
                    "suggestion": suggestion,
                })
 
    def _sentences(self, text: str) -> List[Tuple[int, str]]:
        results = []
        for m in re.finditer(r'[^.!?\n]+[.!?\n]?', text):
            results.append((m.start(), m.group().strip()))
        return results
 
    def analyze(self, text: str, source_file: str = "manual input") -> DetectionResult:
        result    = DetectionResult(original_text=text, source_file=source_file)
        sentences = self._sentences(text)
 
        for entry in self._compiled:
            for match in entry["regex"].finditer(text):
                pos      = match.start()
                sentence = text
                for (start, sent) in sentences:
                    if start <= pos <= start + len(sent):
                        sentence = sent
                        break
                result.issues.append(DetectedIssue(
                    word_found = match.group(),
                    category   = entry["category"],
                    reason     = entry["reason"],
                    suggestion = entry["suggestion"],
                    position   = pos,
                    sentence   = sentence,
                ))
 
        # Deduplicate
        seen, unique = set(), []
        for issue in sorted(result.issues, key=lambda x: x.position):
            key = (issue.position, issue.word_found.lower())
            if key not in seen:
                seen.add(key)
                unique.append(issue)
 
        result.issues         = unique
        result.total_found    = len(unique)
        result.categories_hit = list({i.category for i in unique})
        result.passed         = result.total_found == 0
        result.cleaned_text   = self._auto_fix(text, unique)
        return result
 
    def _auto_fix(self, text: str, issues: List[DetectedIssue]) -> str:
        cleaned = text
        for issue in sorted(issues, key=lambda x: x.position, reverse=True):
            pat = re.compile(re.escape(issue.word_found), re.IGNORECASE)
            cleaned = pat.sub(issue.suggestion if issue.suggestion else " ", cleaned, count=1)
 
        # Post-fix cleanups
        cleaned = re.sub(r'\bis\s+to\s+be\b',   'is',  cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\bwill\s+to\b',       'will', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r' {2,}',               ' ',   cleaned)
        cleaned = cleaned.strip()
        return cleaned
 
    def report(self, result: DetectionResult) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("  LOW CONFIDENCE WORDS — DETECTION REPORT  (v4 — Full Road Coverage)")
        lines.append("=" * 70)
        lines.append(f"\n  Source     : {result.source_file}")
        lines.append(f"  Status     : {result.summary()}")
        lines.append(f"  Issues     : {result.total_found}")
        lines.append(f"  Categories : {len(LOW_CONFIDENCE_PATTERNS)} total coverage")
        if result.categories_hit:
            lines.append(f"  Triggered  : {', '.join(result.categories_hit)}")
 
        if result.issues:
            lines.append("\n" + "-" * 70)
            for i, issue in enumerate(result.issues, 1):
                fix = f"→ '{issue.suggestion}'" if issue.suggestion else "→ REMOVE"
                lines.append(f"\n  #{i}  [{issue.category}]")
                lines.append(f"      Found    : '{issue.word_found}'")
                lines.append(f"      Reason   : {issue.reason}")
                lines.append(f"      Fix      : {fix}")
                lines.append(f"      Sentence : {issue.sentence[:90]}")
 
        lines.append("\n" + "-" * 70)
        lines.append(f"  BEFORE :\n  {result.original_text[:400]}"
                     + ("..." if len(result.original_text) > 400 else ""))
        lines.append(f"\n  AFTER  :\n  {result.cleaned_text[:400]}"
                     + ("..." if len(result.cleaned_text) > 400 else ""))
        lines.append("=" * 70)
        return "\n".join(lines)
 
    def save_report(self, result: DetectionResult, output_path: str = None):
        if output_path is None:
            base = os.path.basename(os.path.splitext(result.source_file)[0])
            output_path = f"{base}_report.txt"
        content = self.report(result) + (
            "\n\n" + "=" * 70 +
            "\n  FULL CORRECTED TEXT\n" + "=" * 70 +
            f"\n\n{result.cleaned_text}\n"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"\n💾 Report saved: {output_path}")
        return output_path
 
 
# ─────────────────────────────────────────────────────────────
# INTERACTIVE MENU
# ─────────────────────────────────────────────────────────────
 
def interactive_menu():
    print("\n" + "=" * 70)
    print("  LOW CONFIDENCE DETECTOR v4 — Full Road Scenario Coverage")
    print(f"  {len(LOW_CONFIDENCE_PATTERNS)} categories | 150+ patterns")
    print("=" * 70)
    print("\n  1. PDF file (.pdf)")
    print("  2. Text file (.txt)")
    print("  3. Type text manually")
    print("  0. Exit\n")
 
    choice = input("  Choice (0-3): ").strip()
    detector = LowConfidenceDetector()
    handler  = FileInputHandler()
 
    if choice == "0":
        sys.exit(0)
    elif choice == "1":
        fp   = input("  PDF path: ").strip().strip('"')
        text = handler.read_pdf(fp);  source = fp
    elif choice == "2":
        fp   = input("  TXT path: ").strip().strip('"')
        text = handler.read_txt(fp);  source = fp
    elif choice == "3":
        text = handler.read_manual(); source = "manual"
    else:
        print("Invalid."); sys.exit(1)
 
    result = detector.analyze(text, source_file=source)
    print(detector.report(result))
 
    if input("\n  Save report? (y/n): ").strip().lower() == "y":
        out = input("  Filename (Enter=auto): ").strip()
        detector.save_report(result, out or None)
 
 
# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Low Confidence Detector v4")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--pdf",  metavar="FILE.pdf")
    g.add_argument("--txt",  metavar="FILE.txt")
    g.add_argument("--text", metavar="'text'")
    parser.add_argument("--save", metavar="out.txt", default=None)
    args = parser.parse_args()
 
    if len(sys.argv) == 1:
        interactive_menu()
        sys.exit(0)
 
    detector = LowConfidenceDetector()
    handler  = FileInputHandler()
 
    if args.pdf:
        text = handler.read_pdf(args.pdf);  source = args.pdf
        result = detector.analyze(text, source)
        print(detector.report(result))
        if args.save: detector.save_report(result, args.save)
    elif args.txt:
        text = handler.read_txt(args.txt);  source = args.txt
        result = detector.analyze(text, source)
        print(detector.report(result))
        if args.save: detector.save_report(result, args.save)
    elif args.text:
        result = detector.analyze(args.text, "command line")
        print(detector.report(result))
        if args.save: detector.save_report(result, args.save)
    else:
        interactive_menu()
