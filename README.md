Lecture 1 - Notes:

  - Threads:
    
    - Lowest granularity of execution
    
    - Executes instructions
    
  - Warps(SIMT):
  
    - Lowest schedulable entity
    
    - Executes instructions in lock-step
    
      - Not every thread needs to execute all instructions
      
  - Threads to Grids
  
    - Thread Blocks:
    
      - Lowest Programmable entity
      
      - Assigned to a single shared core
      
      - Can be 3-D
      
    - Grids:
    
      - The way problem mapped to the GPU
      
      - Part of the GPU launch parameters
      
      - Can be 3-D
