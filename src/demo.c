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
#include <pthread.h>
#include <errno.h>
#include <semaphore.h>

#define DEMO 1

#ifdef OPENCV

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static network *net[60];
static image buff [60];
static image buff_letter[60];
static int buff_index = -1;
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
static pthread_t fetch_thread[60];
static pthread_t display_thread[60];
static sem_t detect_gate[60];

static float fps;
static int every;
static int threads;

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
    int buff_index;
    int net_index;
} detect_input_t;

void *detect_in_thread(void* input_ptr)
{
    running = 1;

    detect_input_t* input = input_ptr;

double t1 = what_time_is_it_now();

    printf("Detecting (%d): %d\n", input->net_index, input->buff_index);


    if (input->run_net) {
        printf("Running net on: %d\n", input->buff_index);

        float nms = .4;

        layer l = net[input->net_index]->layers[net[input->net_index]->n-1];
        float *X = buff_letter[input->buff_index].data;
        network_predict(net[input->net_index], X);

        /*
           if(l.type == DETECTION){
           get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
           } else */
//        remember_network(net);

    //    dets = avg_predictions(net, &nboxes);
        free(dets[input->net_index]);
        dets[input->net_index] = get_network_boxes(net[input->net_index], buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, &nboxes);

        if (nms > 0) do_nms_obj(dets[input->net_index], nboxes, l.classes, nms);

    } else if (input->net_index != input->buff_index) {
        printf("Waiting for finish of detect %d\n", input->net_index * every);
        sem_wait(&detect_gate[input->net_index * every]);
        printf("Finished waiting for finish of detect %d\n", input->net_index * every);
    }

//    printf("\033[2J");
//    printf("\033[1;1H");
//    printf("\nFPS:%.1f\n",fps);
//    printf("Objects:\n\n");
    image display = buff[input->buff_index];
    draw_detections(display, dets[input->net_index], nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);
    demo_index = (demo_index + 1)%demo_frame;

printf("Duration of %d: %f\n", input->buff_index, what_time_is_it_now() - t1);

    if (input->run_net) {
        for (int i = 0; i < (every - 1); ++i) {
            sem_post(&detect_gate[input->buff_index]);
        }
    }

    sem_post(&detect_gate[input->buff_index]);

    free(input);

    running = 0;
    return 0;
}

typedef struct {
    int index;
} fetch_input_t;

void *fetch_in_thread(void *ptr)
{
    fetch_input_t* input = ptr;

        printf("Fetching: %d\n", input->index);

    int status = fill_image_from_stream(cap, buff[input->index]);
    letterbox_image_into(buff[input->index], net[0]->w, net[0]->h, buff_letter[input->index]);
    if(status == 0) demo_done = 1;
    free(input);
    return 0;
}

void *display_in_thread(void *ptr)
{
            printf("Displaying: %d\n", (buff_index)%buff_len);

    show_image_cv(buff[(buff_index )], "Demo", ipl);
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

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen, int every_param, int threads_param)
{
    every = every_param;
    threads = threads_param;
    buff_len = every * threads + 2;
    buff_index = buff_len - 2;
    if (buff_len > sizeof(buff) / sizeof(buff[0]) - 2) error("Too long latency. Decrease -every or -threads.");

    fps = frames != 0 ? frames : 10;

    for (int i = 0; i < threads + 2; ++i) {
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
    for (int i = 0; i < threads + 2; ++i) {
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
        sem_init(&detect_gate[i], 0, 0);
    }

    sem_post(&detect_gate[buff_len - 1]);

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
        double next_step = what_time_is_it_now() + 1.0 / fps;
        int first_buff_index = buff_index;

        for (int j = 0; j < every; ++j) {
            buff_index = (buff_index + 1) % buff_len;
            sleep_until(next_step);

            double t1 = what_time_is_it_now();

            int previous = buff_index;
            int current = (buff_index + 1) % buff_len;
            int next = (buff_index + 2) % buff_len;

            fetch_input_t* fetch_input = malloc(sizeof(fetch_input_t));
            fetch_input->index = next;
            pthread_create(&fetch_thread[next], 0, fetch_in_thread, (void*)fetch_input);
            double t2 = what_time_is_it_now();
            detect_input_t* detect_input = malloc(sizeof(detect_input_t));
            detect_input->buff_index = current;
            detect_input->net_index = current / every;
            detect_input->run_net = current % every == 0;
            pthread_join(fetch_thread[current], NULL);
            if(pthread_create(&detect_thread[current], 0, detect_in_thread, (void*)detect_input)) error("Thread creation failed");
            double t3 = what_time_is_it_now();
            printf("Waiting for detection: %d\n", previous);
            sem_wait(&detect_gate[previous]);
            printf("Done waiting for detection: %d\n", previous);
            display_in_thread(0);
            next_step += 1.0 / fps;
            double now = what_time_is_it_now();
            printf("Done fetching (%.3f), dispatching (%.3f) and displaying (%.3f) %d: %.3f\n\n", t2 - t1, t3 - t2, now - t3, current, now - t1);
        }

        pthread_t t;
        adjust_fps_input_t* adjust_input = malloc(sizeof(adjust_fps_input_t));
        adjust_input->should_be_done_by = next_step;
        adjust_input->every = every;
        adjust_input->index = first_buff_index;

        //if(pthread_create(&t, 0, adjust_fps, (void*)adjust_input)) error("Thread creation failed");
    }
}


#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

