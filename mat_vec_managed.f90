module gpu_kernel
  implicit none
contains
  !------------------
  attributes(global) subroutine matrix_vector_multiply_kernel(A, x, y, N)
    complex*16, device :: A(:,:), x(:), y(:)
    complex*16 :: temp
    integer, value :: N
    integer :: i, j, idx

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
    use cublasxt
    use gpu_kernel
    implicit none

    integer :: N
    integer :: mode
    complex*16, allocatable :: A(:,:), x(:), y(:)
    integer :: i, j
    real :: time0, time1, time2

    print *, 'please input N, mode:'
    read *, N, mode
    print *, 'N:', N, 'mode:', mode

    allocate(A(N, N))
    allocate(x(N))
    allocate(y(N))

    call cpu_time(time0)

    ! initiallize
    call random_seed()
    block
      real(8) :: tmp, tmp2
      do i = 1, N
         call random_number(tmp)
         call random_number(tmp2)
         x(i) = dcmplx(tmp * 2.0d0 - 1.0d0, tmp2 * 2.0d0 - 1.0d0)
         do j = 1, N
            call random_number(tmp)
            call random_number(tmp2)
            A(i, j) = dcmplx(tmp * 2.0d0 - 1.0d0, tmp2 * 2.0d0 - 1.0d0)
         end do
      end do
    end block

    call cpu_time(time1)
    if(mode == 0) then
       ! cublas
       call cublasZgemv('N', N, N, dcmplx(1.0d0, 0.0d0), A, N, x, 1, dcmplx(0.0d0, 0.0d0), y, 1)
       i = cudaDeviceSynchronize()
    else if(mode == 1) then
       ! kernel
       !call matrix_vector_multiply_kernel<<<(N+255)/256, 256>>>(A, x, y, N)
       j = 512
       call matrix_vector_multiply_kernel<<<(N+j-1)/j, j>>>(A, x, y, N)
       i = cudaDeviceSynchronize()
    else if(mode == 2) then
       ! cpu
       y = matmul(A, x)
    end if
    call cpu_time(time2)

    print *, 'Result y(1:10):', y(1:10)

    print *, 'init time:', time1 - time0, 'seconds'
    if(mode == 0) print *, 'CuBlas GPU mat_vec_time (CPU time):', time2 - time1, 'seconds'
    if(mode == 1) print *, 'GPU mat_vec_time (CPU time):', time2 - time1, 'seconds'
    if(mode == 2) print *, 'CPU mat_vec_time (CPU time):', time2 - time1, 'seconds'

end program matrix_vector_multiplication
