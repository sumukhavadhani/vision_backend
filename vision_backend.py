import numpy as np
import cv2
import boto3
import tempfile
import datetime
import pytz
import time

EXTENSION = '.jpg'
MAX_WIDTH = 800

session = boto3.Session(
    region_name='us-east-2',
    aws_access_key_id="<ACCESS_KEY>",
    aws_secret_access_key="<SECRET_KEY>")

dynamodb = session.resource('dynamodb')
s3 = session.client('s3')

visiondb = dynamodb.Table('visiondb')


def list_buckets():
    response = s3.list_buckets()
    buckets = [bucket['Name'] for bucket in response['Buckets']]
    return buckets


def test_bucket_name(buckets, bucket_name):
    return True if bucket_name in buckets else False


def create_bucket(bucket_name):
    s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': 'us-east-2'})


def upload_file_to_s3(filename, bucket_name, key_name):
    s3.upload_file(filename, bucket_name, key_name)


def save_cam_cap_temp(img, filename):
    cv2.imwrite(filename, img)


def get_temp_filename():
    f = tempfile.NamedTemporaryFile(suffix=EXTENSION, delete=False)
    f.close()
    return f.name


def edge_detect(img):
    edges = cv2.Canny(img, 100, 200)
    return edges


def resize_image(img):
    height, width = img.shape[:2]
    if width < MAX_WIDTH:
        new_width = width
    else:
        new_width = MAX_WIDTH
    new_height = int((height * new_width) / width)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_CUBIC)
    return resized_img


def skip_capture_frames_at_start(cap,num_frames=30):
    for i in xrange(num_frames):
        ret, frame = cap.read()
        cv2.imshow('RIMG', frame)
        cv2.waitKey(1)


def update_database(timestamp, rgb_keyname, edge_keyname, year, month, day, utc_epoch_seconds):
    visiondb.put_item(
        Item={
            'timestamp': timestamp,
            'rgb_keyname': rgb_keyname,
            'edge_keyname': edge_keyname,
            'year': year,
            'month': month,
            'day': day,
            'utc_epoch_seconds': utc_epoch_seconds
        }
    )


def get_timstamp_and_epoch_delta():
    u = datetime.datetime.utcnow()
    u = u.replace(tzinfo=pytz.utc)
    e = datetime.datetime.fromtimestamp(0, pytz.utc)
    td = int((u-e).total_seconds())
    return u, td


def camera_capture(max_frames=1000):
    t1 = time.time()
    cap = cv2.VideoCapture(0)
    skip_capture_frames_at_start(cap, num_frames=30)
    bucket_name = datetime.datetime.today().strftime('vision.backend.%Y.%m')
    bucket_names = list_buckets()
    if test_bucket_name(bucket_names, bucket_name) is False:
        create_bucket(bucket_name)
        print("Created S3 bucket: %s" % bucket_name)
    else:
        print("S3 bucket %s exists." % bucket_name)
    temp_rgb_filename = get_temp_filename()
    temp_edge_filename = get_temp_filename()
    print('Using the temp rgb  filename: %s' % temp_rgb_filename)
    print('Using the temp edge filename: %s' % temp_edge_filename)
    frame_count = 0
    t2 = time.time()
    print('Start time delay: %s secs' % (t2-t1,))
    while(True):
        t1 = time.time()
        # Capture frame-by-frame
        ret, frame = cap.read()

        #resize_frame
        rimg = resize_image(frame)

        # convert to gray
        gray = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)

        #edge detect
        edge = edge_detect(gray)

        # save file locally
        cv2.imwrite(temp_rgb_filename, rimg)
        cv2.imwrite(temp_edge_filename, edge)

        # upload to s3
        timestamp, epoch_seconds = get_timstamp_and_epoch_delta()
        timestamp_str = timestamp.strftime('%Y_%m_%d_%H_%M_%S')
        rgb_keyname = timestamp_str + "_rgb" + EXTENSION
        upload_file_to_s3(temp_rgb_filename, bucket_name, rgb_keyname)
        edge_keyname = timestamp_str + "_edge" + EXTENSION
        upload_file_to_s3(temp_edge_filename, bucket_name, edge_keyname)

        update_database(timestamp_str, rgb_keyname, edge_keyname, timestamp.year, timestamp.month, timestamp.day, epoch_seconds)

        frame_count += 1
        t2 = time.time()
        print("frame %s processing and upload complete. Time taken = %s secs." % (frame_count, (t2-t1)))
        # Display the resulting frame
        cv2.imshow('RIMG', rimg)
        cv2.imshow('EDGE', edge)
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (frame_count >= max_frames):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main():
    camera_capture(max_frames=10)


if __name__ == "__main__":
    main()