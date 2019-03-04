# Profiler for any file
'''
Time Profiler:  cProfile
Memory Profiler:

sort_by options
	'nfl' name/file/line
	'ncalls' call count
	'tottime' internal time
	'pcalls' primitive call count
	'cumtime' cumulative time
'''

import cProfile, pstats, io
pr = cProfile.Profile()
pr.enable()
# Run your code here --------------------------------------------------------- #
import experiments.fani
# --------------------------------------------------------------------Profiled #
pr.disable()
s = io.StringIO()
sort_by = 'tottime'
pstats.Stats(pr, stream=s).sort_stats(sort_by).print_stats()
prof = s.getvalue()
with open('prof.txt', 'wb') as f:
	f.write(prof)  # string
#with open("profile.csv", "w") as f:
#	f.write('"ncalls", "tottime", "cumtime", "function", "location"%s' % "\n")
#	for func, (cc, nc, tt, ct, callers) in k.iteritems():
#		f.write('%d, %f, %f, "%s", "%s:%d", %s' % (nc, tt, ct, func[-1], func[0], func[1], "\n"))
