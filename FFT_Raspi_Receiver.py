from bibliopixel.drivers.network_receiver import NetworkReceiver
from bibliopixel.drivers.visualizer import *
from bibliopixel.drivers.LPD8806 import *

from bibliopixel.led import *

coords = [
	[220, 219, 180, 179, 140, 139, 100, 99, 60, 59, 20, 19],
	[221, 218, 181, 178, 141, 138, 101, 98, 61, 58, 21, 18],
	[222, 217, 182, 177, 142, 137, 102, 97, 62, 57, 22, 17],
	[223, 216, 183, 176, 143, 136, 103, 96, 63, 56, 23, 16],
	[224, 215, 184, 175, 144, 135, 104, 95, 64, 55, 24, 15],
	[225, 214, 185, 174, 145, 134, 105, 94, 65, 54, 25, 14],
	[226, 213, 186, 173, 146, 133, 106, 93, 66, 53, 26, 13],
	[227, 212, 187, 172, 147, 132, 107, 92, 67, 52, 27, 12],
	[228, 211, 188, 171, 148, 131, 108, 91, 68, 51, 28, 11],
	[229, 210, 189, 170, 149, 130, 109, 90, 69, 50, 29, 10],
	[230, 209, 190, 169, 150, 129, 110, 89, 70, 49, 30, 9],
	[231, 208, 191, 168, 151, 128, 111, 88, 71, 48, 31, 8],
	[232, 207, 192, 167, 152, 127, 112, 87, 72, 47, 32, 7],
	[233, 206, 193, 166, 153, 126, 113, 86, 73, 46, 33, 6],
	[234, 205, 194, 165, 154, 125, 114, 85, 74, 45, 34, 5],
	[235, 204, 195, 164, 155, 124, 115, 84, 75, 44, 35, 4],
	[236, 203, 196, 163, 156, 123, 116, 83, 76, 43, 36, 3],
	[237, 202, 197, 162, 157, 122, 117, 82, 77, 42, 37, 2],
	[238, 201, 198, 161, 158, 121, 118, 81, 78, 41, 38, 1],
	[239, 200, 199, 160, 159, 120, 119, 80, 79, 40, 39, 0]
]


#must init with same number of pixels as sender
w = 12
h = 20
driver = DriverLPD8806( num = w*h, c_order = ChannelOrder.BRG )
#driver = DriverVisualizer(num=h*w, width = 20, height = 12, stayTop = True)
led = LEDMatrix(driver, width = 12, height = 20, coordMap = coords)
receiver = NetworkReceiver(led)

try:
    receiver.start(join = True) #join = True causes it to not return immediately
except KeyboardInterrupt:
    receiver.stop()
    pass
