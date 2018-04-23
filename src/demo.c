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

static float **predictions;
static float *avg;
static int demo_done = 0;
static int demo_total = 0;

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

typedef enum { false_v, true_v } bool_t;

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);

int size_network(network *net) {
    int count = 0;
    for(int i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            count += l.outputs;
        }
    }
    return count;
}

typedef struct {
    bool_t run_net;
    int buff_index;
    int net_index;
} detect_input_t;

void *detect_in_thread(void* input_ptr) {
    detect_input_t* input = input_ptr;

    pthread_join(fetch_thread[input->buff_index], NULL);

    if (input->run_net) {
        layer l = net[input->net_index]->layers[net[input->net_index]->n-1];
        float *X = buff_letter[input->buff_index].data;
        network_predict(net[input->net_index], X);

        free(dets[input->net_index]);
        dets[input->net_index] = get_network_boxes(net[input->net_index], buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, &nboxes);

        do_nms_obj(dets[input->net_index], nboxes, l.classes, 0.3);

    } else if (input->net_index != input->buff_index) {
        sem_wait(&detect_gate[input->net_index * every]);
    }

    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");
    image display = buff[input->buff_index];
    draw_detections(display, dets[input->net_index], nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);

    if (input->run_net) {
        for (int i = 0; i < (every - 1); ++i) {
            sem_post(&detect_gate[input->buff_index]);
        }
    }

    sem_post(&detect_gate[input->buff_index]);

    free(input);

    return 0;
}

typedef struct {
    int index;
} fetch_input_t;

void *fetch_in_thread(void *ptr) {
    fetch_input_t* input = ptr;

    pthread_join(display_thread[input->index], NULL);

    int status = fill_image_from_stream(cap, buff[input->index]);
    letterbox_image_into(buff[input->index], net[0]->w, net[0]->h, buff_letter[input->index]);
    if(status == 0) demo_done = 1;

    free(input);
    return 0;
}

typedef struct {
    int index;
} display_input_t;

void *display_in_thread(void *ptr) {
    display_input_t* input = ptr;

    double t1 = what_time_is_it_now();
    int sem_val;
    sem_getvalue(&detect_gate[input->index], &sem_val);
    sem_wait(&detect_gate[input->index]);
    double t2 = what_time_is_it_now();

    if (sem_val <= 0) {
        double diff = t2 - t1;
        double n = buff_len;
        double new_fps = 0.95 * n*fps / (n + diff * fps);
        double speed = 0.2 * n / fps;
        if (speed > 1) speed = 1;
        fps = (1 - speed) * fps + speed * new_fps;
    } else {
        double n = every * threads;
        fps += 0.02 * n / fps;
    }

    show_image_cv(buff[(input->index)], "Demo", ipl);
    int c = cvWaitKey(1);
    if (c != -1) c = c%256;
    if (c == 27) {
        demo_done = 1;
        return 0;
    } else if (c == 81) {
        demo_thresh += .02;
    } else if (c == 65) {
        demo_thresh -= .02;
        if(demo_thresh <= .02) demo_thresh = .02;
    } else if (c == 87) {
        demo_hier += .02;
    } else if (c == 83) {
        demo_hier -= .02;
        if(demo_hier <= .0) demo_hier = .0;
    }

    free(input);

    return 0;
}

void sleep_until(double time) {
    double now = what_time_is_it_now();
    double diff = time - now;
    if (diff > 0) {
        usleep(diff * 1000000);
    }
}

void* nullfn(void *ptr) {return 0;}

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen, int every_param, int threads_param)
{
    every = every_param;
    threads = threads_param;
    buff_len = every * threads + 2;
    buff_index = buff_len - 1;
    if (buff_len > sizeof(buff) / sizeof(buff[0]) - 2) error("Too long latency. Decrease -every or -threads.");

    fps = frames != 0 ? frames : 10;

    for (int i = 0; i < threads + 2; ++i) {
        dets[i] = malloc(sizeof(detection));
    }

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

    demo_total = size_network(net[0]);
    predictions = calloc(1, sizeof(float*));
    predictions[1] = calloc(demo_total, sizeof(float));
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

    for (int i = 2; i < buff_len; ++i) {
        sem_post(&detect_gate[i]);
    }

    pthread_create(&fetch_thread[0], 0, nullfn, NULL);
    pthread_create(&display_thread[1], 0, nullfn, NULL);

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

        for (int j = 0; j < every; ++j) {
            buff_index = (buff_index + 1) % buff_len;
            sleep_until(next_step);

            int previous = buff_index;
            int current = (buff_index + 1) % buff_len;
            int next = (buff_index + 2) % buff_len;

            display_input_t* display_input = malloc(sizeof(display_input_t));
            display_input->index = next;
            pthread_create(&display_thread[next], 0, display_in_thread, (void*)display_input);

            fetch_input_t* fetch_input = malloc(sizeof(fetch_input_t));
            fetch_input->index = current;
            pthread_create(&fetch_thread[current], 0, fetch_in_thread, (void*)fetch_input);

            detect_input_t* detect_input = malloc(sizeof(detect_input_t));
            detect_input->buff_index = previous;
            detect_input->net_index = previous / every;
            detect_input->run_net = previous % every == 0;
            pthread_create(&detect_thread[current], 0, detect_in_thread, (void*)detect_input);

            next_step += 1.0 / fps;

        }
    }
}


#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

