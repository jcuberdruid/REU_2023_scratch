# Function to convert string to datetime
def convert(date_time):
	date_time = date_time.split(':')
	hour_seconds = int(date_time[0]) * 60
	seconds_seconds = int(date_time[1]) 
	mS_seconds = int(date_time[2])
	if(mS_seconds > 500): 
		mS_seconds = 1
	else: 
		mS_seconds = 0
	return hour_seconds + seconds_seconds + mS_seconds


test_input_str = "11:15:23"

print(convert(test_input_str))
