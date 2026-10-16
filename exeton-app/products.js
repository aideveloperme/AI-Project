// Sample catalog. Replace names, specs, prices and images with real Exeton data.
window.CATALOG = {
  categories: [
    { id: "all", name: "All" },
    { id: "datacenter", name: "Data Center GPUs" },
    { id: "workstation", name: "Workstation GPUs" },
    { id: "servers", name: "GPU Servers" },
    { id: "networking", name: "Networking" }
  ],
  products: [
    {
      id: "h200-nvl", category: "datacenter", name: "NVIDIA H200 NVL", tag: "New",
      price: 32500, stock: 6, hue: 150,
      summary: "Hopper GPU with 141 GB HBM3e for large language model inference and training.",
      specs: { Architecture: "Hopper", Memory: "141 GB HBM3e", Bandwidth: "4.8 TB/s", "FP8 Tensor": "3,341 TFLOPS", Interface: "PCIe Gen5 x16", "Max Power": "600 W" }
    },
    {
      id: "h100-pcie", category: "datacenter", name: "NVIDIA H100 PCIe 80GB", tag: "Best seller",
      price: 27900, stock: 12, hue: 140,
      summary: "The workhorse for AI training and HPC, with Transformer Engine and NVLink bridge support.",
      specs: { Architecture: "Hopper", Memory: "80 GB HBM2e", Bandwidth: "2.0 TB/s", "FP8 Tensor": "3,026 TFLOPS", Interface: "PCIe Gen5 x16", "Max Power": "350 W" }
    },
    {
      id: "l40s", category: "datacenter", name: "NVIDIA L40S 48GB",
      price: 9200, stock: 20, hue: 175,
      summary: "Universal data center GPU for generative AI, rendering and video workloads.",
      specs: { Architecture: "Ada Lovelace", Memory: "48 GB GDDR6 ECC", Bandwidth: "864 GB/s", "FP8 Tensor": "1,466 TFLOPS", Interface: "PCIe Gen4 x16", "Max Power": "350 W" }
    },
    {
      id: "a100-80", category: "datacenter", name: "NVIDIA A100 80GB",
      price: 14500, stock: 4, hue: 120,
      summary: "Proven Ampere accelerator with Multi-Instance GPU for mixed AI and analytics clusters.",
      specs: { Architecture: "Ampere", Memory: "80 GB HBM2e", Bandwidth: "1.9 TB/s", "TF32 Tensor": "312 TFLOPS", Interface: "PCIe Gen4 x16", "Max Power": "300 W" }
    },
    {
      id: "rtx-pro-6000", category: "workstation", name: "NVIDIA RTX PRO 6000 Blackwell", tag: "New",
      price: 8900, stock: 15, hue: 95,
      summary: "Flagship Blackwell workstation GPU with 96 GB for local AI development and 3D.",
      specs: { Architecture: "Blackwell", Memory: "96 GB GDDR7 ECC", Bandwidth: "1.8 TB/s", Interface: "PCIe Gen5 x16", Outputs: "4x DisplayPort 2.1", "Max Power": "600 W" }
    },
    {
      id: "rtx-6000-ada", category: "workstation", name: "NVIDIA RTX 6000 Ada",
      price: 6800, stock: 9, hue: 105,
      summary: "48 GB professional GPU for design, simulation and AI workstations.",
      specs: { Architecture: "Ada Lovelace", Memory: "48 GB GDDR6 ECC", Bandwidth: "960 GB/s", Interface: "PCIe Gen4 x16", Outputs: "4x DisplayPort 1.4a", "Max Power": "300 W" }
    },
    {
      id: "rtx-4000-ada", category: "workstation", name: "NVIDIA RTX 4000 Ada",
      price: 1250, stock: 40, hue: 85,
      summary: "Single-slot, power-efficient GPU for compact workstations.",
      specs: { Architecture: "Ada Lovelace", Memory: "20 GB GDDR6 ECC", Bandwidth: "360 GB/s", Interface: "PCIe Gen4 x16", Outputs: "4x DisplayPort 1.4a", "Max Power": "130 W" }
    },
    {
      id: "hgx-b200", category: "servers", name: "8x B200 HGX AI Server", tag: "Quote",
      price: null, stock: 2, hue: 200,
      summary: "10U server with NVIDIA HGX B200 8-GPU baseboard for frontier-scale training.",
      specs: { GPUs: "8x NVIDIA B200 SXM", "GPU Memory": "1.44 TB HBM3e", CPUs: "2x Intel Xeon 6", RAM: "Up to 4 TB DDR5", Networking: "8x 400G InfiniBand", "Form Factor": "10U rackmount" }
    },
    {
      id: "hgx-h200", category: "servers", name: "8x H200 HGX AI Server",
      price: null, stock: 3, hue: 210,
      summary: "Production-ready 8U platform with NVLink-connected H200 GPUs.",
      specs: { GPUs: "8x NVIDIA H200 SXM", "GPU Memory": "1.13 TB HBM3e", CPUs: "2x AMD EPYC 9005", RAM: "Up to 3 TB DDR5", Networking: "8x 400G NDR", "Form Factor": "8U rackmount" }
    },
    {
      id: "4u-l40s", category: "servers", name: "4U 8x L40S Inference Server",
      price: 98000, stock: 5, hue: 190,
      summary: "Dense PCIe server for inference, rendering farms and virtual workstations.",
      specs: { GPUs: "8x NVIDIA L40S", "GPU Memory": "384 GB", CPUs: "2x Intel Xeon 5th Gen", RAM: "1 TB DDR5", Storage: "8x 3.84 TB NVMe", "Form Factor": "4U rackmount" }
    },
    {
      id: "cx7", category: "networking", name: "ConnectX-7 400G Adapter",
      price: 2400, stock: 30, hue: 260,
      summary: "Single-port 400 Gb/s InfiniBand / Ethernet adapter for GPU clusters.",
      specs: { Speed: "400 Gb/s", Ports: "1x OSFP", Protocols: "InfiniBand NDR, Ethernet", Interface: "PCIe Gen5 x16", Features: "GPUDirect RDMA" }
    },
    {
      id: "qm9700", category: "networking", name: "Quantum-2 QM9700 Switch",
      price: null, stock: 2, hue: 270,
      summary: "64-port NDR 400G InfiniBand switch for AI fabric spines and leaves.",
      specs: { Ports: "64x 400 Gb/s", Throughput: "51.2 Tb/s", Form: "1U", Management: "Managed (UFM)", Cooling: "Front-to-back" }
    }
  ]
};
