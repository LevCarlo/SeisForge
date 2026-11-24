module hvsr_core
  use, intrinsic :: iso_fortran_env, only: real64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private

  integer, parameter :: dp = real64
  real(dp), parameter :: pi = acos(-1.0_dp)
  real(dp), parameter :: branch_rounding_ulps = 16.0_dp
  ! QUADPACK QK15 abscissae (descending, x >= 0) and Kronrod / Gauss weights,
  ! identical to seisforge.hvsr.body_waves.
  real(dp), parameter :: xgk(8) = [ &
      0.991455371120812639206854697526329_dp, 0.949107912342758524526189684047851_dp, &
      0.864864423359769072789712788640926_dp, 0.741531185599394439863864773280788_dp, &
      0.586087235467691130294144845693013_dp, 0.405845151377397166906606412076961_dp, &
      0.207784955007898467600689403773245_dp, 0.0_dp]
  real(dp), parameter :: wgk(8) = [ &
      0.022935322010529224963732008058970_dp, 0.063092092629978553290700663189204_dp, &
      0.104790010322250183839876322541518_dp, 0.140653259715525918745189590510238_dp, &
      0.169004726639267902826583938629992_dp, 0.190350578064785409913256402421014_dp, &
      0.204432940075298892414161999234649_dp, 0.209482141084727828012999174891714_dp]
  real(dp), parameter :: wg(4) = [ &
      0.129484966168869693270611432679082_dp, 0.279705391489276667901467771423780_dp, &
      0.381830050505118944950369775488975_dp, 0.417959183673469387755102040816327_dp]

  public :: body_wave_contributions, surface_wave_contributions

contains

  pure complex(dp) function vertical_decay_wavenumber(k, omega, velocity) result(gamma)
    ! Re(gamma) >= 0; on the cut (Re(gamma) at the round-off of the argument,
    ! |Re g| |g| <= BRANCH_ROUNDING_ULPS eps (|k|^2 + (omega/v)^2)) Im(gamma) < 0.
    ! Same rule as seisforge.hvsr.propagator.vertical_decay_wavenumber.
    complex(dp), intent(in) :: k
    real(dp), intent(in) :: omega, velocity
    real(dp) :: vertical_squared, rounding

    vertical_squared = (omega / velocity)**2
    gamma = sqrt(k * k - cmplx(vertical_squared, 0.0_dp, dp))
    rounding = branch_rounding_ulps * epsilon(1.0_dp) * (abs(k)**2 + vertical_squared)
    if (abs(real(gamma, dp)) * abs(gamma) <= rounding .and. aimag(gamma) > 0.0_dp) then
      gamma = -gamma
    end if
  end function vertical_decay_wavenumber


  pure subroutine psv_column(k, vp, vs, density, exponent, is_p, column)
    complex(dp), intent(in) :: k, exponent
    real(dp), intent(in) :: vp, vs, density
    logical, intent(in) :: is_p
    complex(dp), intent(out) :: column(4)
    real(dp) :: lambda, mu
    complex(dp), parameter :: imaginary = (0.0_dp, 1.0_dp)

    mu = density * vs**2
    lambda = density * vp**2 - 2.0_dp * mu
    if (is_p) then
      column(1) = imaginary * k
      column(2) = exponent
      column(3) = 2.0_dp * imaginary * mu * k * exponent
      column(4) = (lambda + 2.0_dp * mu) * exponent**2 - lambda * k**2
    else
      column(1) = -exponent
      column(2) = imaginary * k
      column(3) = -mu * (exponent**2 + k**2)
      column(4) = 2.0_dp * imaginary * mu * k * exponent
    end if
  end subroutine psv_column


  pure subroutine psv_layer_bases(thickness, vp, vs, density, omega, k, top, bottom)
    real(dp), intent(in) :: thickness, vp, vs, density, omega
    complex(dp), intent(in) :: k
    complex(dp), intent(out) :: top(4, 4), bottom(4, 4)
    complex(dp) :: gamma_p, gamma_s, exponent(4), state(4)
    real(dp) :: reference
    integer :: column

    gamma_p = vertical_decay_wavenumber(k, omega, vp)
    gamma_s = vertical_decay_wavenumber(k, omega, vs)
    exponent = [-gamma_p, -gamma_s, gamma_p, gamma_s]
    do column = 1, 4
      reference = merge(0.0_dp, thickness, column <= 2)
      call psv_column(k, vp, vs, density, exponent(column), &
                      column == 1 .or. column == 3, state)
      top(:, column) = state * exp(exponent(column) * (0.0_dp - reference))
      bottom(:, column) = state * exp(exponent(column) * (thickness - reference))
    end do
  end subroutine psv_layer_bases


  pure subroutine psv_halfspace_basis(vp, vs, density, omega, k, basis)
    real(dp), intent(in) :: vp, vs, density, omega
    complex(dp), intent(in) :: k
    complex(dp), intent(out) :: basis(4, 2)
    complex(dp) :: gamma_p, gamma_s

    gamma_p = vertical_decay_wavenumber(k, omega, vp)
    gamma_s = vertical_decay_wavenumber(k, omega, vs)
    call psv_column(k, vp, vs, density, -gamma_p, .true., basis(:, 1))
    call psv_column(k, vp, vs, density, -gamma_s, .false., basis(:, 2))
  end subroutine psv_halfspace_basis


  subroutine solve_linear(system, rhs, solution, info)
    complex(dp), intent(in) :: system(:, :), rhs(:, :)
    complex(dp), intent(out) :: solution(size(rhs, 1), size(rhs, 2))
    integer, intent(out) :: info
    complex(dp), allocatable :: a(:, :), b(:, :), row_a(:), row_b(:)
    complex(dp) :: factor
    real(dp) :: pivot_size
    integer :: i, j, pivot, n, nrhs

    n = size(system, 1)
    nrhs = size(rhs, 2)
    allocate(a(n, n), b(n, nrhs), row_a(n), row_b(nrhs))
    a = system
    b = rhs
    info = 0

    do i = 1, n
      pivot = i - 1 + maxloc(abs(a(i:n, i)), dim=1)
      pivot_size = abs(a(pivot, i))
      if (pivot_size <= tiny(1.0_dp)) then
        info = i
        solution = cmplx(0.0_dp, 0.0_dp, dp)
        return
      end if
      if (pivot /= i) then
        row_a = a(i, :)
        a(i, :) = a(pivot, :)
        a(pivot, :) = row_a
        row_b = b(i, :)
        b(i, :) = b(pivot, :)
        b(pivot, :) = row_b
      end if
      do j = i + 1, n
        factor = a(j, i) / a(i, i)
        a(j, i:n) = a(j, i:n) - factor * a(i, i:n)
        b(j, :) = b(j, :) - factor * b(i, :)
      end do
    end do

    solution = cmplx(0.0_dp, 0.0_dp, dp)
    do i = n, 1, -1
      solution(i, :) = (b(i, :) - matmul(a(i, i + 1:n), solution(i + 1:n, :))) / a(i, i)
    end do
  end subroutine solve_linear


  subroutine psv_surface_compliance(thickness, vp, vs, density, frequency, k, compliance, info)
    real(dp), intent(in) :: thickness(:), vp(:), vs(:), density(:), frequency
    complex(dp), intent(in) :: k
    complex(dp), intent(out) :: compliance(2, 2)
    integer, intent(out) :: info
    complex(dp), allocatable :: system(:, :), rhs(:, :), coefficients(:, :)
    complex(dp), allocatable :: tops(:, :, :), bottoms(:, :, :)
    complex(dp) :: halfspace(4, 2), surface_basis(4, 4)
    real(dp) :: omega
    integer :: finite_layers, layer, n_layers, row, unknowns

    n_layers = size(vp)
    finite_layers = n_layers - 1
    unknowns = 4 * finite_layers + 2
    omega = 2.0_dp * pi * frequency
    allocate(system(unknowns, unknowns), rhs(unknowns, 2), coefficients(unknowns, 2))
    system = cmplx(0.0_dp, 0.0_dp, dp)
    rhs = cmplx(0.0_dp, 0.0_dp, dp)

    call psv_halfspace_basis(vp(n_layers), vs(n_layers), density(n_layers), omega, k, halfspace)
    if (finite_layers == 0) then
      surface_basis = cmplx(0.0_dp, 0.0_dp, dp)
      surface_basis(:, 1:2) = halfspace
      system(1:2, 1:2) = halfspace(3:4, :)
    else
      allocate(tops(4, 4, finite_layers), bottoms(4, 4, finite_layers))
      do layer = 1, finite_layers
        call psv_layer_bases(thickness(layer), vp(layer), vs(layer), density(layer), &
                             omega, k, tops(:, :, layer), bottoms(:, :, layer))
      end do
      surface_basis = tops(:, :, 1)
      system(1:2, 1:4) = surface_basis(3:4, :)
    end if
    rhs(1, 1) = 1.0_dp
    rhs(2, 2) = 1.0_dp

    row = 3
    do layer = 1, finite_layers
      system(row:row + 3, 4 * (layer - 1) + 1:4 * layer) = bottoms(:, :, layer)
      if (layer < finite_layers) then
        system(row:row + 3, 4 * layer + 1:4 * (layer + 1)) = -tops(:, :, layer + 1)
      else
        system(row:row + 3, 4 * finite_layers + 1:unknowns) = -halfspace
      end if
      row = row + 4
    end do

    call solve_linear(system, rhs, coefficients, info)
    if (info /= 0) then
      compliance = cmplx(0.0_dp, 0.0_dp, dp)
    else if (finite_layers == 0) then
      compliance = matmul(halfspace(1:2, :), coefficients(1:2, :))
    else
      compliance = matmul(surface_basis(1:2, :), coefficients(1:4, :))
    end if
  end subroutine psv_surface_compliance


  pure subroutine sh_layer_bases(thickness, vs, density, omega, k, top, bottom)
    real(dp), intent(in) :: thickness, vs, density, omega
    complex(dp), intent(in) :: k
    complex(dp), intent(out) :: top(2, 2), bottom(2, 2)
    complex(dp) :: gamma
    real(dp) :: mu

    gamma = vertical_decay_wavenumber(k, omega, vs)
    mu = density * vs**2
    top(:, 1) = [cmplx(1.0_dp, 0.0_dp, dp), -mu * gamma]
    top(:, 2) = [exp(-gamma * thickness), mu * gamma * exp(-gamma * thickness)]
    bottom(:, 1) = [exp(-gamma * thickness), -mu * gamma * exp(-gamma * thickness)]
    bottom(:, 2) = [cmplx(1.0_dp, 0.0_dp, dp), mu * gamma]
  end subroutine sh_layer_bases


  subroutine sh_surface_compliance(thickness, vs, density, frequency, k, compliance, info)
    real(dp), intent(in) :: thickness(:), vs(:), density(:), frequency
    complex(dp), intent(in) :: k
    complex(dp), intent(out) :: compliance
    integer, intent(out) :: info
    complex(dp), allocatable :: system(:, :), rhs(:, :), coefficients(:, :)
    complex(dp), allocatable :: tops(:, :, :), bottoms(:, :, :)
    complex(dp) :: halfspace(2, 1), surface_basis(2, 2), gamma
    real(dp) :: mu, omega
    integer :: finite_layers, layer, n_layers, row, unknowns

    n_layers = size(vs)
    finite_layers = n_layers - 1
    unknowns = 2 * finite_layers + 1
    omega = 2.0_dp * pi * frequency
    gamma = vertical_decay_wavenumber(k, omega, vs(n_layers))
    mu = density(n_layers) * vs(n_layers)**2
    halfspace(:, 1) = [cmplx(1.0_dp, 0.0_dp, dp), -mu * gamma]

    allocate(system(unknowns, unknowns), rhs(unknowns, 1), coefficients(unknowns, 1))
    system = cmplx(0.0_dp, 0.0_dp, dp)
    rhs = cmplx(0.0_dp, 0.0_dp, dp)
    if (finite_layers == 0) then
      surface_basis = cmplx(0.0_dp, 0.0_dp, dp)
      surface_basis(:, 1:1) = halfspace
      system(1, 1) = halfspace(2, 1)
    else
      allocate(tops(2, 2, finite_layers), bottoms(2, 2, finite_layers))
      do layer = 1, finite_layers
        call sh_layer_bases(thickness(layer), vs(layer), density(layer), omega, k, &
                            tops(:, :, layer), bottoms(:, :, layer))
      end do
      surface_basis = tops(:, :, 1)
      system(1, 1:2) = surface_basis(2, :)
    end if
    rhs(1, 1) = 1.0_dp

    row = 2
    do layer = 1, finite_layers
      system(row:row + 1, 2 * (layer - 1) + 1:2 * layer) = bottoms(:, :, layer)
      if (layer < finite_layers) then
        system(row:row + 1, 2 * layer + 1:2 * (layer + 1)) = -tops(:, :, layer + 1)
      else
        system(row:row + 1, 2 * finite_layers + 1:unknowns) = -halfspace
      end if
      row = row + 2
    end do

    call solve_linear(system, rhs, coefficients, info)
    if (info /= 0) then
      compliance = cmplx(0.0_dp, 0.0_dp, dp)
    else if (finite_layers == 0) then
      compliance = halfspace(1, 1) * coefficients(1, 1)
    else
      compliance = surface_basis(1, 1) * coefficients(1, 1) &
                 + surface_basis(1, 2) * coefficients(2, 1)
    end if
  end subroutine sh_surface_compliance


  subroutine body_integrand(thickness_km, vp_km_s, vs_km_s, density_g_cm3, frequency, &
                            k_limit, shift, unit_node, values, info)
    ! k C dk/du for (C_xx, C_zz, C_yy) at one node u.
    real(dp), intent(in) :: thickness_km(:), vp_km_s(:), vs_km_s(:), density_g_cm3(:)
    real(dp), intent(in) :: frequency, k_limit, unit_node
    complex(dp), intent(in) :: shift
    complex(dp), intent(out) :: values(3)
    integer, intent(out) :: info
    complex(dp) :: k, psv(2, 2), sh

    k = k_limit * unit_node * shift
    call psv_surface_compliance(thickness_km, vp_km_s, vs_km_s, density_g_cm3, &
                                frequency, k, psv, info)
    if (info /= 0) then
      info = 2
      return
    end if
    call sh_surface_compliance(thickness_km, vs_km_s, density_g_cm3, frequency, k, sh, info)
    if (info /= 0) then
      info = 3
      return
    end if
    values(1) = psv(1, 1)
    values(2) = psv(2, 2)
    values(3) = sh
    values = values * k * k_limit * shift
  end subroutine body_integrand


  subroutine gauss_kronrod_interval(thickness_km, vp_km_s, vs_km_s, density_g_cm3, &
                                    frequency, k_limit, shift, left, right, kronrod, &
                                    error, info)
    real(dp), intent(in) :: thickness_km(:), vp_km_s(:), vs_km_s(:), density_g_cm3(:)
    real(dp), intent(in) :: frequency, k_limit, left, right
    complex(dp), intent(in) :: shift
    complex(dp), intent(out) :: kronrod(3)
    real(dp), intent(out) :: error(3)
    integer, intent(out) :: info
    complex(dp) :: gauss(3), values(3)
    real(dp) :: half, centre, node, kronrod_weight, gauss_weight
    integer :: j

    half = 0.5_dp * (right - left)
    centre = left + half
    kronrod = (0.0_dp, 0.0_dp)
    gauss = (0.0_dp, 0.0_dp)
    ! Ascending order of the 15 nodes, as in the Python kernel.
    do j = 1, 15
      if (j <= 7) then
        node = -xgk(j)
        kronrod_weight = wgk(j)
      else
        node = xgk(16 - j)
        kronrod_weight = wgk(16 - j)
      end if
      gauss_weight = 0.0_dp
      if (mod(j, 2) == 0) gauss_weight = wg(min(j, 16 - j) / 2)
      call body_integrand(thickness_km, vp_km_s, vs_km_s, density_g_cm3, frequency, &
                          k_limit, shift, centre + half * node, values, info)
      if (info /= 0) return
      kronrod = kronrod + kronrod_weight * values
      gauss = gauss + gauss_weight * values
    end do
    kronrod = half * kronrod
    gauss = half * gauss
    error = abs(kronrod - gauss)
  end subroutine gauss_kronrod_interval


  subroutine body_wave_contributions(thickness_km, vp_km_s, vs_km_s, density_g_cm3, &
                                     frequencies_hz, boundaries, relative_tolerance, &
                                     max_intervals, contour_shift_fraction, &
                                     psv_horizontal, psv_vertical, sh_horizontal, &
                                     relative_error, intervals, converged, status)
    ! Adaptive QK15 on the segments `boundaries` (u = k Vs_N / omega), as in
    ! seisforge.hvsr.body_waves: bisect the interval with the largest
    ! max_c error_c / (rtol |Im I_c|); the left half replaces it, the right
    ! half is appended; stop when every term meets the tolerance or
    ! max_intervals is reached (converged = 0).
    !f2py intent(in) thickness_km, vp_km_s, vs_km_s, density_g_cm3, frequencies_hz
    !f2py intent(in) boundaries, relative_tolerance, max_intervals, contour_shift_fraction
    !f2py intent(out) psv_horizontal, psv_vertical, sh_horizontal
    !f2py intent(out) relative_error, intervals, converged, status
    double precision, intent(in) :: thickness_km(:), vp_km_s(:), vs_km_s(:), density_g_cm3(:)
    double precision, intent(in) :: frequencies_hz(:), boundaries(:)
    double precision, intent(in) :: relative_tolerance, contour_shift_fraction
    integer, intent(in) :: max_intervals
    double precision, intent(out) :: psv_horizontal(size(frequencies_hz))
    double precision, intent(out) :: psv_vertical(size(frequencies_hz))
    double precision, intent(out) :: sh_horizontal(size(frequencies_hz))
    double precision, intent(out) :: relative_error(size(frequencies_hz))
    integer, intent(out) :: intervals(size(frequencies_hz))
    integer, intent(out) :: converged(size(frequencies_hz))
    integer, intent(out) :: status
    real(dp), allocatable :: lefts(:), rights(:), errors(:, :)
    complex(dp), allocatable :: integrals(:, :)
    complex(dp) :: shift, total(3), left_value(3), right_value(3)
    real(dp) :: frequency, k_limit, error(3), target(3), scale(3), middle
    real(dp) :: worst_ratio, ratio
    integer :: frequency_index, count, segment, worst, j, info

    status = 0
    psv_horizontal = 0.0_dp
    psv_vertical = 0.0_dp
    sh_horizontal = 0.0_dp
    relative_error = 0.0_dp
    intervals = 0
    converged = 0
    if (size(vp_km_s) /= size(vs_km_s) .or. size(vp_km_s) /= size(density_g_cm3) .or. &
        size(thickness_km) /= size(vp_km_s) - 1 .or. size(boundaries) < 2 .or. &
        max_intervals < size(boundaries) - 1) then
      status = 1
      return
    end if

    allocate(lefts(max_intervals), rights(max_intervals))
    allocate(integrals(3, max_intervals), errors(3, max_intervals))
    shift = cmplx(1.0_dp, -contour_shift_fraction, dp)

    do frequency_index = 1, size(frequencies_hz)
      frequency = frequencies_hz(frequency_index)
      k_limit = 2.0_dp * pi * frequency / vs_km_s(size(vs_km_s))
      count = size(boundaries) - 1
      do segment = 1, count
        lefts(segment) = boundaries(segment)
        rights(segment) = boundaries(segment + 1)
        call gauss_kronrod_interval(thickness_km, vp_km_s, vs_km_s, density_g_cm3, &
                                    frequency, k_limit, shift, lefts(segment), &
                                    rights(segment), integrals(:, segment), &
                                    errors(:, segment), info)
        if (info /= 0) then
          status = info
          return
        end if
      end do
      do
        total = (0.0_dp, 0.0_dp)
        error = 0.0_dp
        do j = 1, count
          total = total + integrals(:, j)
          error = error + errors(:, j)
        end do
        target = relative_tolerance * abs(aimag(total))
        if (all(error <= target)) then
          converged(frequency_index) = 1
          exit
        end if
        if (count >= max_intervals) exit
        scale = target
        where (scale <= 0.0_dp) scale = tiny(1.0_dp)
        worst = 1
        worst_ratio = -1.0_dp
        do j = 1, count
          ratio = maxval(errors(:, j) / scale)
          if (ratio > worst_ratio) then
            worst_ratio = ratio
            worst = j
          end if
        end do
        middle = 0.5_dp * (lefts(worst) + rights(worst))
        call gauss_kronrod_interval(thickness_km, vp_km_s, vs_km_s, density_g_cm3, &
                                    frequency, k_limit, shift, lefts(worst), middle, &
                                    left_value, error, info)
        if (info /= 0) then
          status = info
          return
        end if
        count = count + 1
        call gauss_kronrod_interval(thickness_km, vp_km_s, vs_km_s, density_g_cm3, &
                                    frequency, k_limit, shift, middle, rights(worst), &
                                    right_value, errors(:, count), info)
        if (info /= 0) then
          status = info
          return
        end if
        lefts(count) = middle
        rights(count) = rights(worst)
        integrals(:, count) = right_value
        rights(worst) = middle
        integrals(:, worst) = left_value
        errors(:, worst) = error
      end do
      intervals(frequency_index) = count
      relative_error(frequency_index) = maxval(error / max(abs(aimag(total)), tiny(1.0_dp)))
      psv_horizontal(frequency_index) = -aimag(total(1)) / 2.0_dp
      psv_vertical(frequency_index) = -aimag(total(2))
      sh_horizontal(frequency_index) = -aimag(total(3)) / 2.0_dp
    end do
  end subroutine body_wave_contributions


  subroutine surface_frequency_function(thickness, vp, vs, density, frequency, k_real, &
                                        is_rayleigh, value, valid)
    real(dp), intent(in) :: thickness(:), vp(:), vs(:), density(:), frequency, k_real
    logical, intent(in) :: is_rayleigh
    real(dp), intent(out) :: value
    logical, intent(out) :: valid
    complex(dp) :: compliance(2, 2), determinant, sh
    integer :: info

    if (is_rayleigh) then
      call psv_surface_compliance(thickness, vp, vs, density, frequency, &
                                  cmplx(k_real, 0.0_dp, dp), compliance, info)
      determinant = compliance(1, 1) * compliance(2, 2) &
                  - compliance(1, 2) * compliance(2, 1)
      if (info /= 0 .or. abs(determinant) <= tiny(1.0_dp)) then
        value = 0.0_dp
        valid = .false.
        return
      end if
      value = real(1.0_dp / determinant, dp)
    else
      call sh_surface_compliance(thickness, vs, density, frequency, &
                                 cmplx(k_real, 0.0_dp, dp), sh, info)
      if (info /= 0 .or. abs(sh) <= tiny(1.0_dp)) then
        value = 0.0_dp
        valid = .false.
        return
      end if
      value = real(1.0_dp / sh, dp)
    end if
    valid = ieee_is_finite(value)
  end subroutine surface_frequency_function


  subroutine bisect_surface_root(thickness, vp, vs, density, frequency, is_rayleigh, &
                                 left_input, right_input, root, success)
    real(dp), intent(in) :: thickness(:), vp(:), vs(:), density(:), frequency
    logical, intent(in) :: is_rayleigh
    real(dp), intent(in) :: left_input, right_input
    real(dp), intent(out) :: root
    logical, intent(out) :: success
    real(dp) :: left, right, middle, f_left, f_right, f_middle
    logical :: valid_left, valid_right, valid_middle
    integer :: iteration

    left = min(left_input, right_input)
    right = max(left_input, right_input)
    call surface_frequency_function(thickness, vp, vs, density, frequency, left, &
                                    is_rayleigh, f_left, valid_left)
    call surface_frequency_function(thickness, vp, vs, density, frequency, right, &
                                    is_rayleigh, f_right, valid_right)
    if (.not. valid_left .or. .not. valid_right .or. f_left * f_right > 0.0_dp) then
      success = .false.
      root = 0.0_dp
      return
    end if

    do iteration = 1, 80
      middle = 0.5_dp * (left + right)
      call surface_frequency_function(thickness, vp, vs, density, frequency, middle, &
                                      is_rayleigh, f_middle, valid_middle)
      if (.not. valid_middle) then
        middle = nearest(middle, right)
        call surface_frequency_function(thickness, vp, vs, density, frequency, middle, &
                                        is_rayleigh, f_middle, valid_middle)
      end if
      if (.not. valid_middle) then
        success = .false.
        root = 0.0_dp
        return
      end if
      if (abs(right - left) <= 1.0e-12_dp * max(1.0_dp, abs(middle))) exit
      if (f_left * f_middle <= 0.0_dp) then
        right = middle
        f_right = f_middle
      else
        left = middle
        f_left = f_middle
      end if
    end do
    root = 0.5_dp * (left + right)
    success = .true.
  end subroutine bisect_surface_root


  subroutine refine_surface_root(thickness, vp, vs, density, frequency, phase_velocity, &
                                 half_width, is_rayleigh, root, success)
    real(dp), intent(in) :: thickness(:), vp(:), vs(:), density(:)
    real(dp), intent(in) :: frequency, phase_velocity, half_width
    logical, intent(in) :: is_rayleigh
    real(dp), intent(out) :: root
    logical, intent(out) :: success
    integer, parameter :: samples = 65
    real(dp) :: omega, low_velocity, high_velocity, k_low, k_high, seed
    real(dp) :: grid(samples), values(samples), candidate, best_distance
    logical :: valid(samples), candidate_success
    integer :: index

    omega = 2.0_dp * pi * frequency
    low_velocity = max(phase_velocity - half_width, 1.0e-8_dp)
    high_velocity = phase_velocity + half_width
    k_low = omega / high_velocity
    k_high = omega / low_velocity
    seed = omega / phase_velocity
    do index = 1, samples
      grid(index) = k_low + real(index - 1, dp) * (k_high - k_low) / real(samples - 1, dp)
      call surface_frequency_function(thickness, vp, vs, density, frequency, grid(index), &
                                      is_rayleigh, values(index), valid(index))
    end do

    success = .false.
    root = 0.0_dp
    best_distance = huge(1.0_dp)
    do index = 1, samples - 1
      if (.not. valid(index) .or. .not. valid(index + 1)) cycle
      if (values(index) * values(index + 1) > 0.0_dp) cycle
      call bisect_surface_root(thickness, vp, vs, density, frequency, is_rayleigh, &
                               grid(index), grid(index + 1), candidate, candidate_success)
      if (candidate_success .and. abs(candidate - seed) < best_distance) then
        root = candidate
        best_distance = abs(candidate - seed)
        success = .true.
      end if
    end do
  end subroutine refine_surface_root


  subroutine add_rayleigh_residue(thickness, vp, vs, density, frequency, root, &
                                  residue_fraction, horizontal, vertical, code)
    ! code: 0 ok, 2 singular boundary system, 3 residue complex or negative
    real(dp), intent(in) :: thickness(:), vp(:), vs(:), density(:)
    real(dp), intent(in) :: frequency, root, residue_fraction
    real(dp), intent(out) :: horizontal, vertical
    integer, intent(out) :: code
    complex(dp) :: plus(2, 2), minus(2, 2), residue(2, 2), h_value, v_value
    real(dp) :: step, tolerance
    integer :: info_plus, info_minus

    vertical = 0.0_dp
    step = max(abs(root) * residue_fraction, 1.0e-10_dp)
    call psv_surface_compliance(thickness, vp, vs, density, frequency, &
                                cmplx(root + step, 0.0_dp, dp), plus, info_plus)
    call psv_surface_compliance(thickness, vp, vs, density, frequency, &
                                cmplx(root - step, 0.0_dp, dp), minus, info_minus)
    horizontal = 0.0_dp
    if (info_plus /= 0 .or. info_minus /= 0) then
      code = 2
      return
    end if
    residue = 0.5_dp * step * (plus - minus)
    h_value = -pi * root * residue(1, 1) / 2.0_dp
    v_value = -pi * root * residue(2, 2)
    tolerance = 1.0e-8_dp * max(abs(h_value), abs(v_value), 1.0_dp)
    if (abs(aimag(h_value)) > tolerance .or. abs(aimag(v_value)) > tolerance .or. &
        real(h_value, dp) < -tolerance .or. real(v_value, dp) < -tolerance) then
      code = 3
      return
    end if
    horizontal = max(real(h_value, dp), 0.0_dp)
    vertical = max(real(v_value, dp), 0.0_dp)
    code = 0
  end subroutine add_rayleigh_residue


  subroutine add_love_residue(thickness, vs, density, frequency, root, residue_fraction, &
                              horizontal, code)
    ! code: 0 ok, 2 singular boundary system, 3 residue complex or negative
    real(dp), intent(in) :: thickness(:), vs(:), density(:)
    real(dp), intent(in) :: frequency, root, residue_fraction
    real(dp), intent(out) :: horizontal
    integer, intent(out) :: code
    complex(dp) :: plus, minus, residue, value
    real(dp) :: step, tolerance
    integer :: info_plus, info_minus

    step = max(abs(root) * residue_fraction, 1.0e-10_dp)
    call sh_surface_compliance(thickness, vs, density, frequency, &
                               cmplx(root + step, 0.0_dp, dp), plus, info_plus)
    call sh_surface_compliance(thickness, vs, density, frequency, &
                               cmplx(root - step, 0.0_dp, dp), minus, info_minus)
    horizontal = 0.0_dp
    if (info_plus /= 0 .or. info_minus /= 0) then
      code = 2
      return
    end if
    residue = 0.5_dp * step * (plus - minus)
    value = -pi * root * residue / 2.0_dp
    tolerance = 1.0e-8_dp * max(abs(value), 1.0_dp)
    if (abs(aimag(value)) > tolerance .or. real(value, dp) < -tolerance) then
      code = 3
      return
    end if
    horizontal = max(real(value, dp), 0.0_dp)
    code = 0
  end subroutine add_love_residue


  subroutine surface_wave_contributions(thickness_km, vp_km_s, vs_km_s, density_g_cm3, &
                                        frequencies_hz, rayleigh_velocity_km_s, &
                                        love_velocity_km_s, velocity_half_width_km_s, &
                                        residue_step_fraction, rayleigh_horizontal, &
                                        rayleigh_vertical, love_horizontal, rayleigh_skip, &
                                        love_skip, status)
    ! rayleigh_skip / love_skip (frequency, mode): 0 included or no seed,
    ! 1 root not refined, 2 singular system at the residue points,
    ! 3 residue complex or negative (seisforge.hvsr.modes.SKIP_REASONS).
    !f2py intent(in) thickness_km, vp_km_s, vs_km_s, density_g_cm3, frequencies_hz
    !f2py intent(in) rayleigh_velocity_km_s, love_velocity_km_s
    !f2py intent(in) velocity_half_width_km_s, residue_step_fraction
    !f2py intent(out) rayleigh_horizontal, rayleigh_vertical, love_horizontal
    !f2py intent(out) rayleigh_skip, love_skip, status
    double precision, intent(in) :: thickness_km(:), vp_km_s(:), vs_km_s(:), density_g_cm3(:)
    double precision, intent(in) :: frequencies_hz(:)
    double precision, intent(in) :: rayleigh_velocity_km_s(:, :), love_velocity_km_s(:, :)
    double precision, intent(in) :: velocity_half_width_km_s, residue_step_fraction
    double precision, intent(out) :: rayleigh_horizontal(size(frequencies_hz))
    double precision, intent(out) :: rayleigh_vertical(size(frequencies_hz))
    double precision, intent(out) :: love_horizontal(size(frequencies_hz))
    integer, intent(out) :: rayleigh_skip(size(rayleigh_velocity_km_s, 1), &
                                          size(rayleigh_velocity_km_s, 2))
    integer, intent(out) :: love_skip(size(love_velocity_km_s, 1), size(love_velocity_km_s, 2))
    integer, intent(out) :: status
    real(dp) :: root, horizontal, vertical, velocity
    logical :: success
    integer :: frequency_index, mode, code

    rayleigh_horizontal = 0.0_dp
    rayleigh_vertical = 0.0_dp
    love_horizontal = 0.0_dp
    rayleigh_skip = 0
    love_skip = 0
    status = 0
    if (size(rayleigh_velocity_km_s, 1) /= size(frequencies_hz) .or. &
        size(love_velocity_km_s, 1) /= size(frequencies_hz)) then
      status = 1
      return
    end if

    do mode = 1, size(rayleigh_velocity_km_s, 2)
      do frequency_index = 1, size(frequencies_hz)
        velocity = rayleigh_velocity_km_s(frequency_index, mode)
        if (.not. ieee_is_finite(velocity) .or. velocity <= 0.0_dp) cycle
        call refine_surface_root(thickness_km, vp_km_s, vs_km_s, density_g_cm3, &
                                 frequencies_hz(frequency_index), velocity, &
                                 velocity_half_width_km_s, .true., root, success)
        if (.not. success) then
          rayleigh_skip(frequency_index, mode) = 1
          cycle
        end if
        call add_rayleigh_residue(thickness_km, vp_km_s, vs_km_s, density_g_cm3, &
                                  frequencies_hz(frequency_index), root, residue_step_fraction, &
                                  horizontal, vertical, code)
        if (code /= 0) then
          rayleigh_skip(frequency_index, mode) = code
          cycle
        end if
        rayleigh_horizontal(frequency_index) = rayleigh_horizontal(frequency_index) + horizontal
        rayleigh_vertical(frequency_index) = rayleigh_vertical(frequency_index) + vertical
      end do
    end do

    do mode = 1, size(love_velocity_km_s, 2)
      do frequency_index = 1, size(frequencies_hz)
        velocity = love_velocity_km_s(frequency_index, mode)
        if (.not. ieee_is_finite(velocity) .or. velocity <= 0.0_dp) cycle
        call refine_surface_root(thickness_km, vp_km_s, vs_km_s, density_g_cm3, &
                                 frequencies_hz(frequency_index), velocity, &
                                 velocity_half_width_km_s, .false., root, success)
        if (.not. success) then
          love_skip(frequency_index, mode) = 1
          cycle
        end if
        call add_love_residue(thickness_km, vs_km_s, density_g_cm3, &
                              frequencies_hz(frequency_index), root, residue_step_fraction, &
                              horizontal, code)
        if (code /= 0) then
          love_skip(frequency_index, mode) = code
          cycle
        end if
        love_horizontal(frequency_index) = love_horizontal(frequency_index) + horizontal
      end do
    end do
  end subroutine surface_wave_contributions

end module hvsr_core
