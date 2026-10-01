import os
import re
import csv
import heapq
from collections import defaultdict
from PIL import Image

# ================= 設定區 =================
DEF_FILE = os.path.join("map", "definition.csv")
BMP_FILE = os.path.join("map", "provinces.bmp")
STATES_DIR = os.path.join("history", "states")
COUNTRIES_DIR = os.path.join("history", "countries")
SUPPLY_NODES_FILE = os.path.join("map", "supply_nodes.txt")
OUTPUT_RAILWAYS = os.path.join("map", "railways.txt")

# ================= 1. 讀取基礎地圖與補給節點 =================
def get_color_map(def_file):
    """讀取 definition.csv，只保留陸地省份的 RGB 對應"""
    color_map = {}
    with open(def_file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if len(row) >= 6:
                # 排除海洋與湖泊
                if row[4].strip().lower() != "land":
                    continue
                try:
                    color_map[(int(row[1]), int(row[2]), int(row[3]))] = row[0]
                except ValueError:
                    pass
    return color_map

def build_adjacency_graph(bmp_file, color_map):
    """掃描 provinces.bmp 建立純陸地相鄰矩陣"""
    graph = defaultdict(set)
    img = Image.open(bmp_file).convert('RGB')
    width, height = img.size
    pixels = img.load()
    
    for y in range(height):
        for x in range(width):
            curr_color = pixels[x, y]
            if x + 1 < width:
                right_color = pixels[x + 1, y]
                if curr_color != right_color:
                    id_curr = color_map.get(curr_color)
                    id_right = color_map.get(right_color)
                    # 只有兩個顏色都是陸地省份，才會建立相鄰關係
                    if id_curr and id_right:
                        graph[id_curr].add(id_right)
                        graph[id_right].add(id_curr)
            if y + 1 < height:
                down_color = pixels[x, y + 1]
                if curr_color != down_color:
                    id_curr = color_map.get(curr_color)
                    id_down = color_map.get(down_color)
                    if id_curr and id_down:
                        graph[id_curr].add(id_down)
                        graph[id_down].add(id_curr)
    return {k: list(v) for k, v in graph.items()}

def get_existing_supply_nodes(supply_file):
    nodes = set()
    if not os.path.exists(supply_file):
        return nodes
    with open(supply_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0] == '1':
                nodes.add(parts[1])
    return nodes

# ================= 2. 解析 States 領土與海軍基地 =================
def parse_states(supply_nodes):
    prov_to_owner = {}
    owner_to_targets = defaultdict(set)
    state_to_provs = {}
    state_to_vps = defaultdict(dict)
    
    for filename in os.listdir(STATES_DIR):
        if not filename.endswith(".txt"):
            continue
            
        filepath = os.path.join(STATES_DIR, filename)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            content = re.sub(r'#.*', '', content)
            
            state_match = re.search(r'id\s*=\s*(\d+)', content)
            owner_match = re.search(r'owner\s*=\s*([A-Za-z0-9_]+)', content)
            if not state_match or not owner_match:
                continue
                
            state_id = state_match.group(1)
            owner = owner_match.group(1)
            
            prov_match = re.search(r'provinces\s*=\s*\{([^\}]+)\}', content)
            if prov_match:
                provs = prov_match.group(1).split()
                state_to_provs[state_id] = provs
                for p in provs:
                    prov_to_owner[p] = owner
                    
            vp_blocks = re.findall(r'victory_points\s*=\s*\{([^\}]+)\}', content)
            for block in vp_blocks:
                tokens = block.split()
                for i in range(0, len(tokens)-1, 2):
                    try:
                        state_to_vps[state_id][tokens[i]] = float(tokens[i+1])
                    except ValueError:
                        continue
                        
            for prov_block in re.finditer(r'(\d+)\s*=\s*\{([^}]+)\}', content):
                if 'naval_base' in prov_block.group(2):
                    owner_to_targets[owner].add(prov_block.group(1))

    for node in supply_nodes:
        if node in prov_to_owner:
            owner = prov_to_owner[node]
            owner_to_targets[owner].add(node)

    return prov_to_owner, owner_to_targets, state_to_provs, state_to_vps

# ================= 3. 解析各國首都 =================
def parse_capitals(state_to_provs, state_to_vps):
    tag_to_capital_prov = {}
    if not os.path.exists(COUNTRIES_DIR):
        return tag_to_capital_prov

    for filename in os.listdir(COUNTRIES_DIR):
        if not filename.endswith(".txt"):
            continue
            
        tag = filename[:3] 
        filepath = os.path.join(COUNTRIES_DIR, filename)
        
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            content = re.sub(r'#.*', '', content)
            
            capital_match = re.search(r'capital\s*=\s*(\d+)', content)
            if capital_match:
                cap_state_id = capital_match.group(1)
                
                if cap_state_id in state_to_provs:
                    vps_in_state = state_to_vps.get(cap_state_id, {})
                    if vps_in_state:
                        best_prov = max(vps_in_state, key=vps_in_state.get)
                        tag_to_capital_prov[tag] = best_prov
                    else:
                        tag_to_capital_prov[tag] = state_to_provs[cap_state_id][0]
                        
    return tag_to_capital_prov

# ================= 4. Dijkstra 尋路與寫入 =================
def find_national_paths(graph, prov_to_owner, start_node, target_nodes, owner_tag):
    queue = [(0, start_node)]
    distances = {start_node: 0}
    came_from = {start_node: None}
    targets_left = set(target_nodes)

    while queue and targets_left:
        current_cost, current_node = heapq.heappop(queue)

        if current_node in targets_left:
            targets_left.remove(current_node)

        for neighbor in graph.get(current_node, []):
            neighbor_owner = prov_to_owner.get(neighbor)
            move_cost = 1 if neighbor_owner == owner_tag else 10
            new_cost = current_cost + move_cost
            
            if neighbor not in distances or new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                came_from[neighbor] = current_node
                heapq.heappush(queue, (new_cost, neighbor))

    paths = {}
    for target in target_nodes:
        if target in came_from:
            path = []
            curr = target
            while curr is not None:
                path.append(curr)
                curr = came_from[curr]
            path.reverse()
            paths[target] = path
        else:
            # 在終端機印出海外孤島的提示，這是正常的
            print(f"    [提示] 節點 {target} 無法從陸地抵達，已略過 (可能是海外領土)。")
            
    return paths

if __name__ == "__main__":
    print("1. 讀取地圖顏色並建立純陸地相鄰矩陣...")
    color_map = get_color_map(DEF_FILE)
    adj_graph = build_adjacency_graph(BMP_FILE, color_map)
    
    print("2. 讀取 map/supply_nodes.txt 中的補給節點...")
    supply_nodes_from_txt = get_existing_supply_nodes(SUPPLY_NODES_FILE)
    
    print("3. 解析 States 尋找海軍基地與分配領土...")
    prov_to_owner, owner_to_targets, state_to_provs, state_to_vps = parse_states(supply_nodes_from_txt)
    
    print("4. 鎖定各國首都...")
    tag_to_capital = parse_capitals(state_to_provs, state_to_vps)
    
    all_railways = []
    
    print("5. 正在鋪設國內鐵路網...")
    for tag, targets in owner_to_targets.items():
        if tag not in tag_to_capital:
            continue
            
        capital_prov = tag_to_capital[tag]
        valid_targets = [t for t in targets if t != capital_prov]
        
        if not valid_targets:
            continue
            
        print(f"  -> 國家 {tag}: 從首都 {capital_prov} 連結至 {len(valid_targets)} 個節點")
        paths = find_national_paths(adj_graph, prov_to_owner, capital_prov, valid_targets, tag)
        
        for path in paths.values():
            if len(path) > 1:
                all_railways.append(f"1 {len(path)} {' '.join(path)}")

    os.makedirs(os.path.dirname(OUTPUT_RAILWAYS), exist_ok=True)
    with open(OUTPUT_RAILWAYS, "w", encoding="utf-8") as f:  # 注意這裡改成 "w" 覆寫，確保你不會和之前生成的錯誤海洋鐵路混在一起
        f.write("# Auto-generated Pure Land Railways (From Capital to nodes & Naval Bases)\n")
        f.write("\n".join(all_railways) + "\n")
        
    print(f"[大功告成] 已成功計算並寫入 {len(all_railways)} 條純陸地鐵路路徑！")