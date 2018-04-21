#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>
#include <unistd.h>

#define DEMO 1

#ifdef OPENCV

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static network *net[60];
static image buff [60];
static image buff_letter[60];
static int buff_index = 0;
static CvCapture * cap;
static IplImage  * ipl;
static float demo_thresh = 0;
static float demo_hier = .5;
static int running = 0;

static int demo_frame = 3;
static int demo_index = 0;
static float **predictions;
static float *avg;
static int demo_done = 0;
static int demo_total = 0;
double demo_time;

static int nboxes = 0;

static int buff_len = 0;
static pthread_t detect_thread[60];

static float fps;
static int every;

static detection *dets[60];

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);

int size_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            count += l.outputs;
        }
    }
    return count;
}

void remember_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(predictions[demo_index] + count, net->layers[i].output, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
}

detection *avg_predictions(network *net, int *nboxes)
{
    int i, j;
    int count = 0;
    fill_cpu(demo_total, 0, avg, 1);
    for(j = 0; j < demo_frame; ++j){
        axpy_cpu(demo_total, 1./demo_frame, predictions[j], 1, avg, 1);
    }
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(l.output, avg + count, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
    detection *dets = get_network_boxes(net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, nboxes);
    return dets;
}

typedef struct {
    bool run_net;
    int index;
} detect_input_t;

void *detect_in_thread(void* input_ptr)
{
    running = 1;

    detect_input_t* input = input_ptr;

    int thread = input->index / every;
double t1 = what_time_is_it_now();
            printf("Detecting: %d\n", input->index);


    if (input->run_net) {
        printf("Running net on: %d\n", input->index);

        float nms = .4;

        layer l = net[thread]->layers[net[thread]->n-1];
        float *X = buff_letter[input->index].data;
        network_predict(net[thread], X);

        /*
           if(l.type == DETECTION){
           get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
           } else */
//        remember_network(net);

    //    dets = avg_predictions(net, &nboxes);
        free(dets[thread]);
        dets[thread] = get_network_boxes(net[thread], buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, &nboxes);

        if (nms > 0) do_nms_obj(dets[thread], nboxes, l.classes, nms);

    }

//    printf("\033[2J");
//    printf("\033[1;1H");
//    printf("\nFPS:%.1f\n",fps);
//    printf("Objects:\n\n");
    image display = buff[input->index];
    draw_detections(display, dets[thread], nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);
    demo_index = (demo_index + 1)%demo_frame;

printf("Duration of %d: %f\n", input->index, what_time_is_it_now() - t1);

    free(input);

    running = 0;
    return 0;
}

void *fetch_in_thread(void *ptr)
{
        printf("Fetching: %d\n", buff_index);

    int status = fill_image_from_stream(cap, buff[buff_index]);
    letterbox_image_into(buff[buff_index], net[0]->w, net[0]->h, buff_letter[buff_index]);
    if(status == 0) demo_done = 1;
    return 0;
}

void *display_in_thread(void *ptr)
{
            printf("Displaying: %d\n", (buff_index + 1)%buff_len);

    show_image_cv(buff[(buff_index + 1)%buff_len], "Demo", ipl);
    int c = cvWaitKey(1);
    if (c != -1) c = c%256;
    if (c == 27) {
        demo_done = 1;
        return 0;
    } else if (c == 82) {
        demo_thresh += .02;
    } else if (c == 84) {
        demo_thresh -= .02;
        if(demo_thresh <= .02) demo_thresh = .02;
    } else if (c == 83) {
        demo_hier += .02;
    } else if (c == 81) {
        demo_hier -= .02;
        if(demo_hier <= .0) demo_hier = .0;
    }
    return 0;
}

void *display_loop(void *ptr)
{
    while(1){
        display_in_thread(0);
    }
}

void *detect_loop(void *ptr)
{
    while(1){
        detect_in_thread(0);
    }
}

void sleep_until(double time) {
    double now = what_time_is_it_now();
    double diff = time - now;
    if (diff > 0) {
        usleep(diff * 1000000);
    }
}

typedef struct {
    double should_be_done_by;
    int index;
    int every;
} adjust_fps_input_t;

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen, int every_param, int threads)
{
    buff_len = every_param * threads;
    every = every_param;
    if (buff_len > sizeof(buff) / sizeof(buff[0])) error("Too long latency. Decrease -every or -threads.");

    fps = frames != 0 ? frames : 10;

    for (int i = 0; i < threads; ++i) {
        dets[i] = malloc(sizeof(detection));
    }

    demo_frame = 1;
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
    for (int i = 0; i < threads; ++i) {
        net[i] = load_network(cfgfile, weightfile, 0);
        set_batch_network(net[i], 1);
    }


    srand(2222222);

    int i;
    demo_total = size_network(net[0]);
    predictions = calloc(demo_frame, sizeof(float*));
    for (i = 0; i < demo_frame; ++i){
        predictions[i] = calloc(demo_total, sizeof(float));
    }
    avg = calloc(demo_total, sizeof(float));

    if(filename){
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);

        if(w){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
        }
        if(h){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
        }
    }

    if(!cap) error("Couldn't connect to webcam.\n");


    for (int i = 0; i < buff_len; ++i) {
        buff[i] = get_image_from_stream(cap);
        buff_letter[i] = letterbox_image(buff[0], net[0]->w, net[0]->h);
    }

    ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

    if(!prefix){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
        if(fullscreen){
            cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        } else {
            cvMoveWindow("Demo", 0, 0);
            cvResizeWindow("Demo", 1352, 1013);
        }
    }

    while(!demo_done){
        double next_step = what_time_is_it_now();
        int first_buff_index = buff_index;

        double t1 = what_time_is_it_now();
        fetch_in_thread(&buff_index);
        detect_input_t* detect_input = malloc(sizeof(detect_input_t));
        detect_input->index = buff_index;
        detect_input->run_net = true;
        if(pthread_create(&detect_thread[buff_index], 0, detect_in_thread, (void*)detect_input)) error("Thread creation failed");
        display_in_thread(0);
        next_step += 1.0 / fps;
        printf("Done fetching, dispatching and displaying %d: %f\n", buff_index, what_time_is_it_now() - t1);

        for (int j = 1; j < every; ++j) {
            buff_index += 1;
            sleep_until(next_step);
            fetch_in_thread(&buff_index);
            detect_input_t* detect_input = malloc(sizeof(detect_input_t));
            detect_input->index = buff_index;
            detect_input->run_net = false;
            if(pthread_create(&detect_thread[buff_index], 0, detect_in_thread, (void*)detect_input)) error("Thread creation failed");
            display_in_thread(0);
            next_step += 1.0 / fps;
        }

        pthread_t t;
        adjust_fps_input_t* input = malloc(sizeof(adjust_fps_input_t));
        input->should_be_done_by = next_step;
        input->every = every;
        input->index = first_buff_index;

        //if(pthread_create(&t, 0, adjust_fps, (void*)input)) error("Thread creation failed");
        buff_index = (buff_index + 1) % buff_len;
        sleep_until(next_step);
    }
}


#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

