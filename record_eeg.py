import sys
import os
import numpy as np
import time

sys.path.append('C:/Users/Balint/Documents/gtec/Unicorn Suite/Hybrid Black/Unicorn Python/Lib') # Add path to the unicorn folder
import UnicornPy

def connect_to_device():
    deviceList = UnicornPy.GetAvailableDevices(True)

    if len(deviceList) <= 0 or deviceList is None:
        raise Exception("No device available. Please pair with a Unicorn first.")

    print("Available devices:")
    for i, device in enumerate(deviceList):
        print("#%i %s" % (i, device))

    deviceID = 0
    if deviceID < 0 or deviceID > len(deviceList):
        raise IndexError('The selected device ID is not valid.')

    print("Trying to connect to '%s'." % deviceList[deviceID])
    device = UnicornPy.Unicorn(deviceList[deviceID])
    print("Connected to '%s'." % deviceList[deviceID])

    return device

def record_data(device_set, duration=60, frame_length=1):
    device = device_set
    TestsignaleEnabled = False
    all_data = bytearray()
    numberOfAcquiredChannels = device.GetNumberOfAcquiredChannels()
    receiveBufferBufferLength = frame_length * numberOfAcquiredChannels * 4

    device.StartAcquisition(TestsignaleEnabled)
    print("Data acquisition started.")

    numberOfGetDataCalls = int(duration * UnicornPy.SamplingRate / frame_length)
    consoleUpdateRate = int((UnicornPy.SamplingRate / frame_length) / 25.0)
    if consoleUpdateRate == 0:
        consoleUpdateRate = 1

    for i in range(0, numberOfGetDataCalls):
        receiveBuffer = bytearray(receiveBufferBufferLength)
        device.GetData(frame_length, receiveBuffer, receiveBufferBufferLength)
        all_data.extend(receiveBuffer)  # Append data to the bytearray

        if i % consoleUpdateRate == 0:
            print('.', end='', flush=True)

    device.StopAcquisition()
    print("\nData acquisition stopped.")

    # Convert bytearray to a flat numpy array of float32 values
    flat_data = np.frombuffer(all_data, dtype=np.float32)
    
    # Reshape the flat array into a 2D array
    reshaped_data = flat_data.reshape(-1, numberOfAcquiredChannels)
    
    return reshaped_data