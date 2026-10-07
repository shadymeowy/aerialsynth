// Headless GPU bring-up check: adapter info and the limits the backend relies on.
fn main() -> anyhow::Result<()> {
    let g = render::gpu::device::Gpu::new()?;
    println!("{:?}", g.info);
    let l = g.device.limits();
    println!(
        "max_texture_array_layers {} max_texture_dimension_2d {} max_storage_buffer_binding_size {} max_buffer_size {} max_compute_invocations {}",
        l.max_texture_array_layers, l.max_texture_dimension_2d, l.max_storage_buffer_binding_size, l.max_buffer_size, l.max_compute_invocations_per_workgroup
    );
    Ok(())
}
