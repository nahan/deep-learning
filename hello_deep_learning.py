from datetime import datetime

print('Hello Deep Learning!')

start_datetime = datetime(2018, 1, 24, 15, 00)

today_datetime = datetime.now()

diff = today_datetime - start_datetime

format_str = 'You have been practicing Deep Learning for: \n%s days, %s hours, %s minutes, %s seconds. \nKeep Running!'

days = diff.days
hours = diff.seconds / 3600
minutes = (diff.seconds - (hours * 3600)) / 60
seconds = diff.seconds - (hours * 3600) - (minutes * 60)

print(format_str % (days, hours, minutes, seconds))

