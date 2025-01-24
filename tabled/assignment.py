from typing import List

import numpy as np
from surya.schema import TableResult, Bbox

from tabled.heuristics import heuristic_layout
from tabled.schema import SpanTableCell

def y_overlap_pct(c1, c2, y_margin=0):
    overlap = max(0, min(c1.bbox[3] + y_margin, c2.bbox[3] + y_margin) - max(c1.bbox[1] - y_margin, c2.bbox[1] - y_margin))
    c1_height = c1.bbox[3] - c1.bbox[1]
    c2_height = c2.bbox[3] - c2.bbox[1]
    max_height = max(c1_height, c2_height)
    return (overlap / max_height) * 100 if max_height > 0 else 0

def x_overlap_pct(c1, c2, x_margin=0):
    overlap = max(0, min(c1.bbox[2] + x_margin, c2.bbox[2] + x_margin) - max(c1.bbox[0] - x_margin, c2.bbox[0] - x_margin))
    c1_width = c1.bbox[2] - c1.bbox[0]
    c2_width = c2.bbox[2] - c2.bbox[0]
    max_width = max(c1_width, c2_width)
    return (overlap / max_width) * 100 if max_width > 0 else 0

def is_rotated(rows, cols):
    # Determine if the table is rotated by looking at row and column width / height ratios
    # Rows should have a >1 ratio, cols <1
    widths = sum([r.width for r in rows])
    heights = sum([c.height for c in rows]) + 1
    r_ratio = widths / heights

    widths = sum([c.width for c in cols])
    heights = sum([r.height for r in cols]) + 1
    c_ratio = widths / heights

    return r_ratio * 2 < c_ratio


def overlapper_idxs(rows, field, thresh=.3):
    overlapper_rows = set()
    for row in rows:
        row_id = getattr(row, field)
        if row_id in overlapper_rows:
            continue

        for row2 in rows:
            row2_id = getattr(row2, field)
            if row2_id == row_id or row2_id in overlapper_rows:
                continue

            if row.intersection_pct(row2) > thresh:
                i_bigger = row.area > row2.area
                overlapper_rows.add(row_id if i_bigger else row2_id)
    return overlapper_rows


def initial_assignment(detection_result: TableResult, thresh=.5) -> List[SpanTableCell]:
    print(detection_result)
    overlapper_rows = overlapper_idxs(detection_result.rows, field="row_id")
    overlapper_cols = overlapper_idxs(detection_result.cols, field="col_id")

    print("OverLaps: \n")
    print(overlapper_rows)
    print(overlapper_cols)

    cells = []
    for cell in detection_result.cells:
        max_intersection = 0
        row_pred = None
        for row in detection_result.rows:
            if row.row_id in overlapper_rows:
                continue

            intersection_pct = y_overlap_pct(cell,row)
            if intersection_pct > max_intersection and intersection_pct > thresh:
                max_intersection = intersection_pct
                row_pred = row.row_id

        max_intersection = 0
        col_pred = None
        for col in detection_result.cols:
            if col.col_id in overlapper_cols:
                continue

            intersection_pct = x_overlap_pct(cell,col)
            if intersection_pct > max_intersection and intersection_pct > thresh:
                max_intersection = intersection_pct
                col_pred = col.col_id

        cells.append(
            SpanTableCell(
                bbox=cell.bbox,
                text=cell.text,
                row_ids=[row_pred],
                col_ids=[col_pred]
            )
        )
    return cells


def assign_overlappers(cells: List[SpanTableCell], detection_result: TableResult, thresh=.5):
    overlapper_rows = overlapper_idxs(detection_result.rows, field="row_id")
    overlapper_cols = overlapper_idxs(detection_result.cols, field="col_id")

    for cell in cells:
        max_intersection = 0
        row_pred = None
        for row in detection_result.rows:
            if row.row_id not in overlapper_rows:
                continue

            intersection_pct = y_overlap_pct(cell,row)
            if intersection_pct > max_intersection and intersection_pct > thresh:
                max_intersection = intersection_pct
                row_pred = row.row_id

        max_intersection = 0
        col_pred = None
        for col in detection_result.cols:
            if col.col_id not in overlapper_cols:
                continue

            intersection_pct = x_overlap_pct(cell,col)
            if intersection_pct > max_intersection and intersection_pct > thresh:
                max_intersection = intersection_pct
                col_pred = col.col_id

        if cell.row_ids[0] is None:
            cell.row_ids = [row_pred]
        if cell.col_ids[0] is None:
            cell.col_ids = [col_pred]


def assign_unassigned(table_cells: list, detection_result: TableResult):
    rotated = is_rotated(detection_result.rows, detection_result.cols)
    for cell in table_cells:
        if cell.row_ids[0] is None:
            closest_row = None
            min_dist = None
            for row in detection_result.rows:
                if rotated:
                    dist = cell.center_x_distance(row)
                else:
                    dist = cell.center_y_distance(row)

                if min_dist is None or dist < min_dist:
                    closest_row = row.row_id
                    min_dist = dist
            cell.row_ids = [closest_row]

        if cell.col_ids[0] is None:
            closest_col = None
            min_dist = None
            for col in detection_result.cols:
                if rotated:
                    dist = cell.center_y_distance(col)
                else:
                    dist = cell.center_x_distance(col)

                if min_dist is None or dist < min_dist:
                    closest_col = col.col_id
                    min_dist = dist

            cell.col_ids = [closest_col]


def handle_rowcol_spans(table_cells: list, detection_result: TableResult, thresh=.25):
    rotated = is_rotated(detection_result.rows, detection_result.cols)
    for cell in table_cells:
        for c in detection_result.cols:
            col_intersect_pct = cell.intersection_y_pct(c) if rotated else cell.intersection_x_pct(c)
            other_cell_exists = len([tc for tc in table_cells if tc.col_ids[0] == c.col_id and tc.row_ids[0] == cell.row_ids[0]]) > 0
            if col_intersect_pct > thresh and not other_cell_exists:
                cell.col_ids.append(c.col_id)
            else:
                break
        # Assign to first column header appears in
        cell.col_ids = sorted(cell.col_ids)

    for cell in table_cells:
        for r in detection_result.rows:
            row_intersect_pct = cell.intersection_x_pct(r) if rotated else cell.intersection_y_pct(r)
            other_cell_exists = len([tc for tc in table_cells if tc.row_ids[0] == r.row_id and tc.col_ids[0] == cell.col_ids[0]]) > 0
            if row_intersect_pct > thresh and not other_cell_exists:
                cell.row_ids.append(r.row_id)
            else:
                break


def merge_multiline_rows(detection_result: TableResult, table_cells: List[SpanTableCell]):
    def find_row_gap(r1, r2):
        return min([abs(r1.bbox[1] - r2.bbox[3]), abs(r2.bbox[1] - r1.bbox[3])])

    def calculate_box_distance(box1, box2):
        """
        Calcule la distance verticale entre deux boîtes si elles se chevauchent horizontalement,
        sinon, calcule la distance entre les coins les plus proches.
    
        :param box1: Liste [x1, y1, x2, y2] pour la première boîte
        :param box2: Liste [x1, y1, x2, y2] pour la deuxième boîte
        :return: Distance entre les deux boîtes
        """
        # Vérifier si les boîtes se chevauchent horizontalement
        if box1[2] >= box2[0] and box2[2] >= box1[0]:
            # Les boîtes se chevauchent horizontalement
            if box1[3] < box2[1]:  # box1 est au-dessus de box2
                vertical_distance = box2[1] - box1[3]
            elif box2[3] < box1[1]:  # box2 est au-dessus de box1
                vertical_distance = box1[1] - box2[3]
            else:
                vertical_distance = 0  # Les boîtes se touchent ou se chevauchent verticalement
            return vertical_distance
        else:
            # Les boîtes ne se chevauchent pas horizontalement
            # Calculer la distance entre les coins les plus proches
            if box1[2] < box2[0]:  # box1 est complètement à gauche de box2
                closest_distance = min(
                    ((box1[2] - box2[0]) ** 2 + (box1[1] - box2[3]) ** 2) ** 0.5,  # Coin inférieur droit de box1 vers supérieur gauche de box2
                    ((box1[2] - box2[0]) ** 2 + (box1[3] - box2[1]) ** 2) ** 0.5   # Coin supérieur droit de box1 vers inférieur gauche de box2
                )
            elif box2[2] < box1[0]:  # box2 est complètement à gauche de box1
                closest_distance = min(
                    ((box2[2] - box1[0]) ** 2 + (box2[1] - box1[3]) ** 2) ** 0.5,  # Coin inférieur droit de box2 vers supérieur gauche de box1
                    ((box2[2] - box1[0]) ** 2 + (box2[3] - box1[1]) ** 2) ** 0.5   # Coin supérieur droit de box2 vers inférieur gauche de box1
                )
            else:
                closest_distance = 0  # Cas où il y a une erreur logique (ne devrait pas arriver)
            return closest_distance
        
    def find_gap_cells(list_cell1,list_cell2):
        values = []
        for c1 in list_cell1:
            for c2 in list_cell2:
                values.append(calculate_box_distance(c1.bbox,c2.bbox))
        return min(values)

    all_cols = set([tc.col_ids[0] for tc in table_cells])
    if len(all_cols) == 0:
        return

    merged_pairs = []
    row_gaps = [
        find_row_gap(r, r2)
        for r, r2 in zip(detection_result.rows, detection_result.rows[1:])
    ]
    if len(row_gaps) == 0:
        return
    gap_thresh = np.percentile(row_gaps,80)
    nb_row = len(detection_result.rows)
    if nb_row>0:
        idx = 0
        new_rows = [detection_result.rows[0]]
        current_cells = []
        for idx in range(nb_row):
            prev_row = new_rows[-1]
            row = detection_result.rows[idx]
            # Ensure the gap between r2 and r1 is small

            current_cells.extend([tc for tc in table_cells if tc.row_ids[0] == idx-1])
            r1_cells = current_cells
            r2_cells = [tc for tc in table_cells if tc.row_ids[0] == row.row_id]
            r1_cols = set([tc.col_ids[0] for tc in r1_cells])
            r2_cols = set([tc.col_ids[0] for tc in r2_cells])
            if len(r1_cells)>0 and len(r2_cells)>0:
                gap = find_gap_cells(r1_cells,r2_cells)
            else :
                gap = find_row_gap(prev_row, row)
            print(f"Gap : {gap}")

            if len(r2_cells) == 0:
                continue
            
    
            # Ensure all columns in r2 are in r1
            print(f"R1_Col = {r1_cols}")
            print(f"R2_Col = {r2_cols}")
            print(f"R1_Cel = {r1_cells}")
            print(f"R2_Cel = {r2_cells}\n\n")
            if len(r2_cells) == 0:
                continue

            
            reasons = []

            if gap > gap_thresh:
                reasons.append("'gap > gap_thresh'")
            
           # if len(r2_cols - r1_cols) > 0:
           #     reasons.append("'len(r2_cols - r1_cols) > 0'")
            
            #if len(r2_cols) / len(all_cols) > 0.9:
            #    reasons.append("'len(r2_cols) / len(all_cols) > 0.9'")
            
            if reasons:  # Si au moins une condition est vraie
                print("Raisons pour entrer dans la boucle :", ", ".join(reasons))
          
                new_rows[-1].bbox = [
                    min([c.bbox[0] for c in current_cells]+ [new_rows[-1].bbox[0]]),
                    min([c.bbox[1] for c in current_cells]+ [new_rows[-1].bbox[1]]),
                    max([c.bbox[2] for c in current_cells]+ [new_rows[-1].bbox[2]]),
                    max([c.bbox[3] for c in current_cells]+ [new_rows[-1].bbox[3]])
                ]
                new_rows.append(row)
                current_cells = []
                continue
            current_cells.extend([tc for tc in table_cells if tc.row_ids[0] == idx-1])                
            r2_idx = row.row_id
            new_rows[-1].bbox = [
                min([c.bbox[0] for c in current_cells]+ [new_rows[-1].bbox[0]]),
                min([c.bbox[1] for c in current_cells]+ [new_rows[-1].bbox[1]]),
                max([c.bbox[2] for c in current_cells]+ [new_rows[-1].bbox[2]]),
                max([c.bbox[3] for c in current_cells]+ [new_rows[-1].bbox[3]])
            ]
            print("Merged !")
            print(new_rows[-1].bbox)

        for i in range(len(new_rows)):
            new_rows[i].row_id=i
            
        detection_result.rows = new_rows
        print(new_rows)


def assign_rows_columns(detection_result: TableResult, image_size: list, heuristic_thresh=.6) -> List[SpanTableCell]:
    table_cells = initial_assignment(detection_result, thresh = 0.6)
    assign_unassigned(table_cells, detection_result)
    print("Table:")
    print(table_cells)
    merge_multiline_rows(detection_result, table_cells)
    print("Table Merged:")
    print(table_cells)
    table_cells = initial_assignment(detection_result,thresh = 0.6)
    print("Table reassign:")
    print(table_cells)
    #assign_overlappers(table_cells, detection_result, thresh = 0.6)
    assign_unassigned(table_cells, detection_result)

    #print("Table overlaps:")
    print(table_cells)
    total_unassigned = len([tc for tc in table_cells if tc.row_ids[0] is None or tc.col_ids[0] is None])
    print(f"Non assigné {total_unassigned}. \n")
    unassigned_frac = total_unassigned / max(len(table_cells), 1)

    if unassigned_frac > heuristic_thresh:
        table_cells = heuristic_layout(table_cells, image_size)
        return table_cells

    assign_unassigned(table_cells, detection_result)
    handle_rowcol_spans(table_cells, detection_result)
    return table_cells
