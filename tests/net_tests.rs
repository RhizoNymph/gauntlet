//! Two-sided TCP peer tests over loopback.

use gauntlet::agent::net;

async fn spawn_server(port: u16) -> tokio::task::JoinHandle<anyhow::Result<()>> {
    let handle = tokio::spawn(net::serve(port));
    // Give the listener a beat to bind before clients dial in.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    handle
}

#[tokio::test]
async fn latency_probe_reports_a_distribution() {
    let port = 39501;
    let server = spawn_server(port).await;

    let report = net::measure_latency(&format!("127.0.0.1:{port}"), 1)
        .await
        .expect("latency probe");
    assert!(
        report.samples > 100,
        "loopback for 1s: got {}",
        report.samples
    );
    assert!(report.rtt_p50_us > 0.0);
    assert!(
        report.rtt_p50_us <= report.rtt_p99_us && report.rtt_p99_us <= report.rtt_max_us,
        "distribution must be ordered: {report:?}"
    );
    assert!(
        report.rtt_p99_us < 10_000.0,
        "loopback p99 above 10ms is broken plumbing: {report:?}"
    );

    net::shutdown_peer(&format!("127.0.0.1:{port}"))
        .await
        .expect("shutdown");
    server.await.expect("join").expect("server exits cleanly");
}

#[tokio::test]
async fn bandwidth_probe_moves_real_bytes() {
    let port = 39502;
    let server = spawn_server(port).await;

    let report = net::measure_bandwidth(&format!("127.0.0.1:{port}"), 1)
        .await
        .expect("bandwidth probe");
    assert!(
        report.bytes_sent > 10 << 20,
        "1s on loopback must move >10 MiB, got {}",
        report.bytes_sent
    );
    assert!(
        report.gib_per_sec > 0.01 && report.gib_per_sec.is_finite(),
        "implausible loopback throughput: {report:?}"
    );

    net::shutdown_peer(&format!("127.0.0.1:{port}"))
        .await
        .expect("shutdown");
    server.await.expect("join").expect("server exits cleanly");
}

#[tokio::test]
async fn serve_handles_sequential_clients() {
    let port = 39503;
    let server = spawn_server(port).await;
    let target = format!("127.0.0.1:{port}");

    let first = net::measure_latency(&target, 1)
        .await
        .expect("first client");
    let second = net::measure_bandwidth(&target, 1)
        .await
        .expect("second client");
    assert!(first.samples > 0);
    assert!(second.bytes_sent > 0);

    net::shutdown_peer(&target).await.expect("shutdown");
    server.await.expect("join").expect("server exits cleanly");
}

#[tokio::test]
async fn client_errors_cleanly_when_no_peer() {
    // Nothing listens here; expect an error, not a hang (bounded by connect
    // failing fast on loopback).
    let result = net::measure_latency("127.0.0.1:39599", 1).await;
    assert!(result.is_err());
}
