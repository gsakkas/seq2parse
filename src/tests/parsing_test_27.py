def calculate_blame_in_line(full_schedule, valid_bd_events, line, scorer):
    i = 0
    j = 0
    rogue_trains = () # might contain duplicates
    time = get_time_event(valid_bd_event)
    loc_id = get_location_id_in_line(bd_event, line)
    while i < len(valid_bd_events):
            rogue_trains = rogue_trains + set(tuple(get_rogue_schedules_in_line(full_schedule, line, time[i], loc_id[i])),)
            i += 1
            while j < len(rogue_trains):
                    scorer = blame_train(scorer,rogue_trains[j][0])
                    j += 1
            return scorer

