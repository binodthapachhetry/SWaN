import calendar, json, os, sys, time
from SWaN-accel import _root



def _get_dataset_folder():
    if os.path.exists(_root.relpath('datasets_custom')):
        return 'datasets_custom'
    return 'datasets'

def _get_labels_folder():
    if os.path.exists(_root.relpath('labels_custom')):
        return 'labels_custom'
    return 'labels'

def datasetDir(dataset):
    return _root.relpath(_get_dataset_folder(), dataset)

def datasetTileDir(dataset):
    return _root.relpath(_get_dataset_folder(), dataset, 'tiles')

def datasetConfigFilename(dataset):
    return _root.relpath(_get_dataset_folder(), dataset, 'config.json')

def latestLabelsFilename(dataset, session):
    return _root.relpath(_get_labels_folder(), dataset, session, 'labels.latest.json')

def logLabelsFilename(dataset, session):
    return _root.relpath(_get_labels_folder(), dataset, session, 'labels.log.jsons')

def exportFilename(dataset):
    return _root.relpath('export', dataset + '.csv')



def getLabelsSessions(dataset):
    sessions = []

    folder = _root.relpath(_get_labels_folder(), dataset)
    if os.path.exists(folder):
        for fn in os.listdir(folder):
            sessions.append(fn)

    return sessions

def getLabelsLatest(dataset):
    sessions = getLabelsSessions(dataset)

    labelsall = []
    for session in sessions:
        labelfilename = latestLabelsFilename(dataset, session)
        if os.path.exists(labelfilename):
            with open(labelfilename, 'rb') as lfile:
                labels = json.loads(lfile.read())
                labelsall.append(labels)
    return labelsall



def ensureDirExists(name, isFile):
    if isFile:
        dirname = os.path.dirname(name)
    else:
        dirname = name
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return name



def activityJSON(a, wasprev):
    comma = ''
    if wasprev:
        comma = ','
    return comma + ('{"lo":%d, "hi":%d, "label":"%s"}' % (a[0], a[1], a[2]))



def timeMillisecondToTimeString(ms):
    sec = ms / 1000
    msec = ms % 1000

    tm = time.gmtime(sec)
    return time.strftime('%Y-%m-%d %H:%M:%S', tm) + ('.%03d' % msec)

def timeStringToTimeMillisecond(tm):
    parts = tm.split('.')
    if len(parts) == 1:
        sec = parts[0]
        msec = '000'
    elif len(parts) == 2:
        sec = parts[0]
        msec = parts[1]
        if len(msec) > 3:
            print('*** ERROR: time more detailed than milliseconds: ' + tm)
            sys.exit(-1)
        while len(msec) < 3:
            msec = msec + '0'
    else:
        print('*** ERROR: invalid time format: ' + tm)
        sys.exit(-1)

    parsed = None
    for fmt in ['%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S']:
        try:
            parsed = time.strptime(sec, fmt)
            break
        except:
            pass

    if not parsed:
        print('*** ERROR: unparseable time format: ' + tm)
        sys.exit(-1)

    return 1000 * int(calendar.timegm(parsed)) + int(msec)
