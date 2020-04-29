####################
# osr_msgs track message
####################
# ID
uint8 id

# Type (1: PERSON, 2: CAR)
uint8 type

# Posture (1: STAND, 2: SIT DOWN, 3: LIE DOWN)
uint8 posture

# Activity score
# float32 activity

# Bounding box
osr_msgs/BoundingBox bbox_pose
osr_msgs/BoundingBox bbox_velocity

# State
geometry_msgs/Pose pose
geometry_msgs/Twist twist

# 3D Bounding Box

