from bibliopixel.drivers.network_receiver import NetworkReceiver
from bibliopixel.drivers.LPD8806 import *

from bibliopixel.led import *

#must init with same number of pixels as sender
w = 12
h = 20
driver = DriverLPD8806( num = w*h, c_order = ChannelOrder.BRG )
#driver = DriverVisualizer(num=h*w, width = 20, height = 12, stayTop = True)
led = LEDMatrix(driver, width = 12, height = 20)
receiver = NetworkReceiver(led)

try:
    receiver.start(join = True) #join = True causes it to not return immediately
except KeyboardInterrupt:
    receiver.stop()
    pass
