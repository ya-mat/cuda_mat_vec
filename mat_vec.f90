module gpu_kernel
  implicit none
contains
  !------------------
  attributes(global) subroutine matrix_vector_multiply_kernel(A, x, y, N)
    real(4), device :: A(:,:), x(:), y(:)
    integer, value :: N
    integer :: i, j, idx
    real(4) :: temp
 
    idx = threadIdx%x + (blockIdx%x - 1) * blockDim%x
    if (idx < N) then
       temp = 0.0
       do j = 1, N
          temp = temp + A(idx, j) * x(j)
       end do
       y(idx) = temp
    end if
  end subroutine matrix_vector_multiply_kernel
  !------------------
end module gpu_kernel

program matrix_vector_multiplication
    use cudafor
    use gpu_kernel
    implicit none

    integer :: N
    real(4), allocatable :: A(:,:), x(:), y(:)
    real(4), device, allocatable :: d_A(:, :), d_x(:), d_y(:)
    integer :: i, j
    real :: start_time, to_device_time, kernel_time, to_host_time, mat_vec_start_time, mat_vec_end_time

    print *, 'please input N:'
    read *, N

    allocate(A(N, N))
    allocate(x(N))
    allocate(y(N))
    allocate(d_A(N, N))
    allocate(d_x(N))
    allocate(d_y(N))

    ! 初期化
    A = reshape([((real(i * N + j, kind=4), j = 1, N), i = 1, N)], shape(A))
    x = [(real(i, kind=4), i = 1, N)]

    call cpu_time(start_time)

    ! デバイスに転送
    d_A = A
    d_x = x

    call cpu_time(to_device_time)

    ! カーネル呼び出し
    call matrix_vector_multiply_kernel<<<(N+255)/256, 256>>>(d_A, d_x, d_y, N)
    i = cudaDeviceSynchronize()

    call cpu_time(kernel_time)

    ! 結果をホストに転送
    y = d_y

    call cpu_time(to_host_time)

    print *, 'Result y(1:10):', y(1:10)

    call cpu_time(mat_vec_start_time)

    y = matmul(A, x)

    call cpu_time(mat_vec_end_time)

    print *, 'Result y(1:10):', y(1:10)

    print *, 'to_device_time (CPU time):', to_device_time - start_time, 'seconds'
    print *, 'kernel_time (CPU time):', kernel_time - to_device_time, 'seconds'
    print *, 'to_host_time (CPU time):', to_host_time - kernel_time, 'seconds'
    print *, 'GPU mat_vec_time (CPU time):', to_host_time - start_time, 'seconds'
    print *, 'CPU mat_vec_time (CPU time):', mat_vec_end_time - mat_vec_start_time, 'seconds'

end program matrix_vector_multiplication
