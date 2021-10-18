####################
# osr_msgs annotation message
####################
# Header
Header header

# 2D Bounding Box
#BoundingBox bbox
uint32 x
uint32 y
uint32 width
uint32 height

# ID
int32 id

# Pose
std_msgs/String pose

# Class
std_msgs/String cls

# Modal
std_msgs/String modal
