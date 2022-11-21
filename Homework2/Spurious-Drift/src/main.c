#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

const double x0 = 0;
const double dt = 0.01;
const double sigma0 = 1;
const double delta_sigma = 1.8;

const int N = 10000;
const double T = 100000;
const double L = 100;
const double ALPHA = 1;

void dumpCsv(FILE* fptr, double* x) {
    for ( int i = 0; i < N; i++ ) {
        fprintf(fptr, "%f,", x[i]);
    }
    fprintf(fptr, "\n");
}

void setDirs(float* dirs) {
    for ( int i = 0; i < N; i++ ) {
        dirs[i] = rand() % 2 * 2 - 1;
    }
}

void getNowAsDateStr(char* text, int textLen) {
    time_t now = time(NULL);
    struct tm *t = localtime(&now);

    /* Leave one byte for null-termination */
    strftime(text, textLen-1, "./output/%d_%m_%Y_%H-%M-%S.csv", t);
}

double sigma(double x) {
    return sigma0 + delta_sigma/L * x;
}

double dsigma(double x) {
    return delta_sigma/L;
}

void simulateTrajectories(double* x, int n) {
    const double dt_sqrt = sqrt(dt);

    for ( int i = 0; i < n; i++ ) {
        x[i] = x0;
    }

    double t = 0;
    int j = 0;

    srand(time(NULL));
    float diff[n];

    while ( t < T ) {
        setDirs(diff);
        for ( int i = 0; i < n; i++ ) {
            diff[i] *= sigma(x[i]) * dt_sqrt;

            /* Noise-induced drift */
            x[i] += ALPHA * sigma(x[i]) * dsigma(x[i]) * dt;

            x[i] += diff[i];

            if ( x[i] < -L/2 ) {
                x[i] = -L - x[i];
            } else if ( x[i] > L/2 ) {
                x[i] = L - x[i];
            }
        }

        if ( j % 10000 == 0 ) {
            printf("j: %d, t: %f/%f\n", j, t, T);
        }

        t += dt;
        j++;
    }
}

int main() {
    double x[N];

    simulateTrajectories(x, N);

    char dateString[100];
    getNowAsDateStr(dateString, sizeof(dateString));

    FILE* fptr;
    fptr = fopen(dateString, "w");

    dumpCsv(fptr, x);

    return 0;
}
