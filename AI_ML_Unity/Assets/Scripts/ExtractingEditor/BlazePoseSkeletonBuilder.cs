using System.Collections;
using System.Collections.Generic;
using System.IO;
using System;
using UnityEngine;
using UnityEditor;
using UnityEngine.SceneManagement;
using UnityEditor.SceneManagement;
using UnityEngine.TextCore.Text;
using UnityEngine.VFX;

public class BlazePoseSkeletonBuilder : ScriptableObject
{
    public Actor actor;
    public Dictionary<int, int> boneMap = new Dictionary<int, int>(); 
    
    // BlazePose 연결 정의
    public int[,] blazePoseBones = new int[,] {
        {11, 12}, {11, 23}, {12, 24}, {23, 24},
        {11, 13}, {13, 15}, {15, 17}, {15, 19}, {15, 21},
        {12, 14}, {14, 16}, {16, 18}, {16, 20}, {16, 22},
        {23, 25}, {25, 27}, {27, 29}, {29, 31},
        {24, 26}, {26, 28}, {28, 30}, {30, 32},
        {0, 1}, {1, 2}, {2, 3}, {3, 7},
        {0, 4}, {4, 5}, {5, 6}, {6, 8},
        {9, 10}, {0, 9}, {0, 10},
        //{11, 0}, {12, 0}
    };

    public void BuildActorSkeleton()
    {
        GameObject root = new GameObject("blaze_pose_actor");
        Transform[] joints = new Transform[33];
        for (int i = 0; i < 33; i++)
        {
            GameObject joint = new GameObject($"joint_{i}");
            joint.transform.parent = root.transform;
            joint.transform.localPosition = Vector3.zero;
            joint.transform.localRotation = Quaternion.identity;
            joints[i] = joint.transform;
        }

        // parent-child 연결
        for (int i = 0; i < blazePoseBones.GetLength(0); i++)
        {
            int parent = blazePoseBones[i, 0];
            int child = blazePoseBones[i, 1];
            joints[child].parent = joints[parent];
            joints[child].localPosition = Vector3.up * 0.05f; // 일정한 길이
            joints[child].localRotation = Quaternion.identity;
        }

        // Actor에 적용
        actor = root.AddComponent<Actor>();
        actor.ExtractSkeleton(joints); // Transform 계층을 기반으로 Bone[] 구성
        BuildMapping(actor);
        Debug.Log($"✅ Created procedural Actor skeleton with {actor.Bones.Length} bones");
    }

    // 초기 1회 매핑 생성
    public void BuildMapping(Actor actor)
    {
        boneMap.Clear();
        for (int i = 0; i < 33; i++)
        {
            Actor.Bone b = Array.Find(actor.Bones, x => x.GetName() == $"joint_{i}");
            if (b != null)
            {
                boneMap[i] = b.Index;
            }
            else
            {
                Debug.LogWarning($"⚠️ joint_{i} not found in actor.");
            }
        }
        Debug.Log($"✅ Bone mapping table built: {boneMap.Count} entries");
    }


    
}
