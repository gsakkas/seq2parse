def friendsOfFriends(d):
    for key1 in d:
        for key2 in d:
            if key1 == key2:
                continue
            else:
                d[key1].update(d[key1]- d[key2])
    return d

print(friendsOfFriends({"fred":
    set(["wilma", "betty", "barney", "bam-bam"]),
    "wilma": set(["fred", "betty", "dino"])}))

