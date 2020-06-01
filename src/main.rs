use vulkano::{
    instance::{
        Instance,
        PhysicalDevice,
    },

    pipeline::{
        GraphicsPipeline, 
        viewport::Viewport, 
		vertex::{
			SingleBufferDefinition
		}
    },

    device::{
        Device,
        Features,
        RawDeviceExtensions,
    },

    framebuffer::{
        Framebuffer, 
        FramebufferAbstract, 
        Subpass, 
        RenderPassAbstract
    },

    image::{
        SwapchainImage, 
    },

    buffer::{
        BufferUsage,
        CpuAccessibleBuffer,
    },

    command_buffer::{
        AutoCommandBufferBuilder, 
        DynamicState
    },

    swapchain,
    swapchain::{
        ColorSpace,
        FullscreenExclusive,
        AcquireError, 
        Swapchain, 
        SurfaceTransform, 
        PresentMode, 
        SwapchainCreationError
    },

    sync, 
    sync::{
        GpuFuture, 
        FlushError
    },
};

use std::sync::Arc;

use vulkano_win::VkSurfaceBuild; 
use winit::{
    event_loop::{
        ControlFlow, 
        EventLoop, 
    },
    window::{
        Window, 
        WindowBuilder, 
    },
    event::{
        Event, 
        WindowEvent
    }
};


#[derive(Default, Debug, Clone, Copy)]
struct Vertex2dTex {
	position: [f32; 2],
	uv: [f32; 2],
}
vulkano::impl_vertex!(Vertex2dTex, position, uv); 

fn main() {
    let instance = {
        let inst_exts = vulkano_win::required_extensions(); 
        Instance::new(None, &inst_exts, None).expect("failed to create instance")
    };

    let physical = PhysicalDevice::enumerate(&instance)
        .next()
        .expect("no device available");

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics())
        .expect("couldn't find a graphical queue family");

    let (device, mut queues) = {
        let unraw_dev_exts = vulkano::device::DeviceExtensions {
            khr_swapchain: true, 
            .. vulkano::device::DeviceExtensions::none()
        };
        let mut dev_exts = RawDeviceExtensions::from(&unraw_dev_exts);
        dev_exts.insert(std::ffi::CString::new("VK_KHR_storage_buffer_storage_class").unwrap());


        let dev_features = Features {
            geometry_shader: true, 
            .. Features::none()
        };

        Device::new(
            physical,
            &dev_features, 
            dev_exts,
            [(queue_family, 0.5)].iter().cloned(),
        )
        .expect("failed to create device")
    };

    let queue = queues.next().unwrap();

    let event_loop = EventLoop::new(); 

    let surface = WindowBuilder::new().build_vk_surface(&event_loop, instance.clone()).unwrap();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical) 
            .expect("failed to get surface capabilities"); 
        let usage = caps.supported_usage_flags; 
        let alpha = caps.supported_composite_alpha.iter().next().unwrap(); 
        let format = caps.supported_formats[0].0;

        let dim: [u32; 2] = surface.window().inner_size().into();

        Swapchain::new(
            device.clone(), 
            surface.clone(), 
            caps.min_image_count, format, dim, 1, usage, &queue, 
            SurfaceTransform::Identity, alpha, PresentMode::Fifo, FullscreenExclusive::Default, false, ColorSpace::SrgbNonLinear)
        .expect("failed to create swapchain")
    };

    let render_pass = Arc::new(vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    ).unwrap());

    mod vs { 
        vulkano_shaders::shader!{
            ty: "vertex", 
            path: "./src/vs.glsl"
        }
    }
    #[allow(dead_code)] // Used to force recompilation of shader change
    const VS: &str = include_str!("./vs.glsl");
    let vs = vs::Shader::load(device.clone()).unwrap(); 

    mod fs { 
        vulkano_shaders::shader!{
            ty: "fragment", 
            path: "./src/fs.glsl"
        }
    }
    #[allow(dead_code)] // Used to force recompilation of shader change
    const FS: &str = include_str!("./fs.glsl");
    let fs = fs::Shader::load(device.clone()).unwrap(); 

    let pipeline = Arc::new(GraphicsPipeline::start()
        .vertex_input(SingleBufferDefinition::<Vertex2dTex>::new())
        .vertex_shader(vs.main_entry_point(), ())
        .triangle_strip()
        .viewports_dynamic_scissors_irrelevant(1)
        .fragment_shader(fs.main_entry_point(), ())
        .blend_alpha_blending()
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap()
    );

    let vertex_buffer = {
        CpuAccessibleBuffer::from_iter(
            device.clone(), 
            BufferUsage::all(), 
            false, 
            [
                Vertex2dTex { position: [-0.5, -0.5], uv: [0.0, 0.0] },
                Vertex2dTex { position: [-0.5,  0.5], uv: [0.0, 1.0] },
                Vertex2dTex { position: [ 0.5, -0.5], uv: [1.0, 0.0] },
                Vertex2dTex { position: [ 0.5,  0.5], uv: [1.0, 1.0] },
            ].iter().cloned()
        ).unwrap();
    };

    let mut dynamic_state = DynamicState { 
        line_width: None, 
        viewports: None, 
        scissors: None, 
        compare_mask: None, 
        write_mask: None, 
        reference: None 
    }; 
    
    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state); 

    let mut recreate_swapchain = false; 

    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>); 

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => *control_flow = ControlFlow::Exit,
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => recreate_swapchain = true,
            Event::RedrawEventsCleared => {
                previous_frame_end.as_mut().unwrap().cleanup_finished(); 

                if recreate_swapchain {
                    let dim: [u32; 2] = surface.window().inner_size().into();
                    let (new_swapchain, new_images) = match swapchain.recreate_with_dimensions(dim) {
                        Ok(r) => r, 
                        Err(SwapchainCreationError::UnsupportedDimensions) => return, 
                        Err(err) => panic!("failed to recreate swapchain {:?}", err)
                    }; 

                    swapchain = new_swapchain; 
                    framebuffers = window_size_dependent_setup(&new_images, render_pass.clone(), &mut dynamic_state); 
                    recreate_swapchain = false; 
                }

                let (image_num, suboptimal, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None){
                    Ok(r) => r, 
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true; 
                        return; 
                    }, 
                    Err(err) => panic!("{:?}", err)
                }; 

                if suboptimal {
                    recreate_swapchain = true; 
                }

                let clear_values = vec!([1.0, 1.0, 1.0, 1.0].into()); 
                let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(
                    device.clone(), 
                    queue.family()
                )
                    .unwrap()
                    .begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
                    .unwrap()
                    .draw(
                        pipeline.clone(), 
                        &dynamic_state, 
                        vertex_buffer.clone(), 
                        (), 
                        ()
                    )
                    .unwrap()
                    .end_render_pass()
                    .unwrap()
                    .build()
                    .unwrap();

                let future = previous_frame_end.take().unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer).unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num) 
                    .then_signal_fence_and_flush(); 

                match future {
                    Ok(future) => {
                        future.wait(None).unwrap(); 
                        previous_frame_end = Some(Box::new(future) as Box<_>);
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true; 
                        previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                    }
                    Err(e) => {
                        println!("{:?}", e);
                        previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>); 
                    }
                }
            },
            _ => ()
        }
    });
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>], 
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>, 
    dynamic_state: &mut DynamicState
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions(); 

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32], 
        depth_range: 0.0 .. 1.0, 
    }; 

    dynamic_state.viewports = Some(vec!(viewport)); 

    images.iter().map(|image| {
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(image.clone()).unwrap()
                .build().unwrap()
        ) as Arc<dyn FramebufferAbstract + Send + Sync>
    }).collect::<Vec<_>>()
}

