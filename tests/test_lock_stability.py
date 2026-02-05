import random
from lib.lock_logic import Candidate, LockState, compute_dynamic_radius, select_locked_candidate, update_lock_state


def test_reacquire_after_minor_occlusion():
    state = LockState(
        x=100,
        y=100,
        last_seen_time=0.0,
        last_seen_x=100,
        last_seen_y=100,
        last_area=1000.0,
        last_update_time=0.0,
    )
    candidate = Candidate(
        abs_x=220,
        abs_y=100,
        rel_x=0,
        rel_y=0,
        x1=0,
        y1=0,
        x2=50,
        y2=80,
        conf=0.7,
        crosshair_dist=0.0,
        area=900.0,
    )
    selected, _ = select_locked_candidate(
        (candidate,),
        state,
        now=0.2,
        base_radius=80,
        reacquire_radius=150,
        grace_time=0.25,
        min_conf=0.4,
        min_area_ratio=0.4,
        ping_ms=60,
        ping_scale=0.6,
    )
    assert selected is candidate


def test_release_after_grace_expires():
    state = LockState(
        x=100,
        y=100,
        last_seen_time=0.0,
        last_seen_x=100,
        last_seen_y=100,
        last_area=1000.0,
        last_update_time=0.0,
    )
    candidate = Candidate(
        abs_x=320,
        abs_y=100,
        rel_x=0,
        rel_y=0,
        x1=0,
        y1=0,
        x2=50,
        y2=80,
        conf=0.7,
        crosshair_dist=0.0,
        area=900.0,
    )
    selected, _ = select_locked_candidate(
        (candidate,),
        state,
        now=1.0,
        base_radius=80,
        reacquire_radius=150,
        grace_time=0.25,
        min_conf=0.4,
        min_area_ratio=0.4,
        ping_ms=60,
        ping_scale=0.6,
    )
    assert selected is None


def test_ping_compensation_expands_radius():
    state = LockState(vel_x=300.0, vel_y=400.0)
    radius = compute_dynamic_radius(state, base_radius=80.0, ping_ms=100.0, ping_scale=1.0)
    assert radius == 130.0


def test_stress_lock_with_occlusion_and_teleport():
    rng = random.Random(7)
    state = LockState()
    now = 0.0
    dt = 1 / 60
    base_radius = 80.0
    reacquire_radius = 150.0
    grace_time = 0.25
    min_conf = 0.4
    min_area_ratio = 0.4
    ping_ms = 60.0
    ping_scale = 0.6

    true_x = 100.0
    true_y = 100.0
    vx = 120.0
    vy = 30.0
    drops = 0

    for i in range(300):
        now += dt
        if 90 <= i < 100:
            candidates = ()
        else:
            if i == 180:
                true_x += 320.0
            else:
                true_x += vx * dt
                true_y += vy * dt
            jitter_x = rng.randint(-3, 3)
            jitter_y = rng.randint(-3, 3)
            candidates = (
                Candidate(
                    abs_x=int(true_x + jitter_x),
                    abs_y=int(true_y + jitter_y),
                    rel_x=0,
                    rel_y=0,
                    x1=0,
                    y1=0,
                    x2=50,
                    y2=80,
                    conf=0.7,
                    crosshair_dist=0.0,
                    area=4000.0,
                ),
            )

        selected, _ = select_locked_candidate(
            candidates,
            state,
            now=now,
            base_radius=base_radius,
            reacquire_radius=reacquire_radius,
            grace_time=grace_time,
            min_conf=min_conf,
            min_area_ratio=min_area_ratio,
            ping_ms=ping_ms,
            ping_scale=ping_scale,
        )

        if selected is not None:
            update_lock_state(state, selected, now)
        else:
            update_lock_state(state, None, now)
            if state.x is not None and (now - state.last_seen_time) > grace_time:
                state = LockState()
                drops += 1
            if state.x is None and candidates:
                update_lock_state(state, candidates[0], now)

    assert drops == 1
