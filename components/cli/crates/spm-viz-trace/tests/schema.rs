use spm_viz_trace::{ExpertEvent, RouteEvent, RoutingTrace, Trace, next_frame};

fn event() -> ExpertEvent {
    ExpertEvent {
        layer: 2,
        expert: 7,
        selected: true,
        routed_tokens: 3,
        packed_bytes: 1_314_816,
        decoded_bytes: 6_291_456,
        layer_read_us: 14_000,
        layer_decode_us: 9_000,
        layer_compute_us: 60_000,
    }
}

#[test]
fn schema_names_measurement_scope_and_every_required_field() {
    let mut trace = Trace {
        model: "granite-test",
        schedule: "all-expert",
        events: Vec::new(),
    };
    trace.push(event()).expect("bounded event");
    let json = trace.to_json();
    for field in [
        "emufpga.serial-moe.v1",
        "measured-layer-total",
        "\"layer\":2",
        "\"expert\":7",
        "\"selected\":true",
        "\"routed_tokens\":3",
        "\"packed_bytes\":1314816",
        "\"decoded_bytes\":6291456",
        "\"layer_read_us\":14000",
        "\"layer_decode_us\":9000",
        "\"layer_compute_us\":60000",
    ] {
        assert!(json.contains(field), "missing {field}");
    }
}

#[test]
fn animation_wraps_only_while_playing() {
    assert_eq!(next_frame(3, 8, false), 3);
    assert_eq!(next_frame(3, 8, true), 4);
    assert_eq!(next_frame(7, 8, true), 0);
    assert_eq!(next_frame(0, 0, true), 0);
}

#[test]
fn trace_refuses_more_than_one_event_per_expert_and_layer() {
    let mut trace = Trace {
        model: "granite-test",
        schedule: "all-expert",
        events: Vec::new(),
    };
    for _ in 0..Trace::MAX_EVENTS {
        trace.push(event()).expect("within bound");
    }
    assert!(trace.push(event()).is_err());
}

#[test]
fn routing_trace_has_a_bounded_inert_schema() {
    let mut trace = RoutingTrace {
        model: "granite-test",
        events: Vec::new(),
    };
    trace
        .push(RouteEvent {
            layer: 2,
            token: 7,
            experts: [1, 3, 5, 7, 9, 11, 13, 15],
        })
        .unwrap();
    assert_eq!(
        trace.to_json(),
        "{\"schema\":\"emufpga.moe-routing.v1\",\"model\":\"granite-test\",\"events\":[{\"layer\":2,\"token\":7,\"experts\":[1,3,5,7,9,11,13,15]}]}"
    );
    assert!(
        trace
            .push(RouteEvent {
                layer: 0,
                token: 0,
                experts: [32; 8]
            })
            .is_err()
    );
}
