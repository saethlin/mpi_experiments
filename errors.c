#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>


void print_error_code(int code) {
    char message[MPI_MAX_ERROR_STRING + 1] = {0};
    int len;
    if (!MPI_Error_string(code, message, &len)) {
        fprintf(stderr, "%s, error code %d\n", message, code);
    } else {
        fprintf(stderr, "No error message known for error code %d\n", code);
    }
}


int const COMPUTE = 0;
int const SHUTDOWN = 1;
int const RETURNING_DATA = 2;
int rank, n_ranks;

#define NUM_DATA 1000


int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    MPI_Init(&argc, &argv);
    // This is actually a whole other aspect I need to work on
    //MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    // Pick a rank at random to kill
    srand(time(NULL));
    int dead_rank = (rand() % (n_ranks - 1)) + 1;


    if (rank == 0) {
        MPI_Request* requests = malloc(n_ranks * sizeof(MPI_Request));
        int data[NUM_DATA] = {0};
        int num_data_collected = 0;

        // Send an initial message to every rank
        for (int r = 1; r < n_ranks; r++) {
            MPI_Isend(NULL, 0, MPI_INT, r, 0, MPI_COMM_WORLD, &requests[r]);
        }

        // Main message loop
        while (1) {
            MPI_Recv(&data[num_data_collected], 1, MPI_INT, MPI_ANY_SOURCE,
                    RETURNING_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int from_rank = data[num_data_collected];
            num_data_collected += 1;

            if (num_data_collected < NUM_DATA) { // If we're not done, send another request
                MPI_Isend(NULL, 0, MPI_INT, from_rank, 0, MPI_COMM_WORLD, &requests[from_rank]);
            } else {
                break;
            }
        }
        printf("collected all the data\n");
        for (int i = 0; i < NUM_DATA; i++) {
            printf("%d, ", data[i]);
        }
        printf("\n");

        // Send shutdown requests
        for (int r = 1; r < n_ranks; r++) {
            // Do everything we can to destroy the existing request(s)
            MPI_Cancel(&requests[r]);
            MPI_Request_free(&requests[r]);

            // Send a shutdown request
            MPI_Isend(NULL, 0, MPI_INT, r, SHUTDOWN, MPI_COMM_WORLD, &requests[r]);
        }

        printf("sent all the shutdown requests\n");
        sleep(1);
        for (int r = 1; r < n_ranks; r++) {
            int completed;
            MPI_Status status;
            MPI_Test(&requests[r], &completed, &status);
            if (!completed) {
                printf("rank %d shutdown send did not complete\n", r);
            }
        }

        printf("shutting down\n\n");
    } else if (rank == dead_rank) {
        abort();
    } else {
        int data = rank;
        MPI_Status status;
        MPI_Request request = MPI_REQUEST_NULL;
        int send_attempts = 0;

        while (1) {
            MPI_Recv(NULL, 0, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == SHUTDOWN) {
                MPI_Cancel(&request);
                MPI_Request_free(&request);
                printf("rank %d exiting, send attempts: %d\n", rank, send_attempts);
                break;
            }
            // Sleep for 10ms as if we're doing some calculation
            usleep(10000);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            MPI_Isend(&data, 1, MPI_INT, 0, RETURNING_DATA, MPI_COMM_WORLD, &request);
            send_attempts += 1;
        }
    }
}
