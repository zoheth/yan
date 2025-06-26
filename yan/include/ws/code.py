CollectiveMainLoop collective_main_loop
CollectiveEpilogue collective_epilogue

if producer:
    reg_dealloc(LoadRegisterRequirement)
    work_tile = get_initial_work()
    while(work_tile.is_valid()):
        auto block_coord = work_tile_info.get_block_coord()
        collective_main_loop.load(block_coord)
        work_tile = get_next_work()
else:
    
