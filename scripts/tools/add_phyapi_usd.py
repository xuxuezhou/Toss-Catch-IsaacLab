from pxr import Usd, UsdGeom, UsdPhysics, Gf

def analyze_usd_file(file_path):

    # Open the USD stage
    stage = Usd.Stage.Open(file_path)
    if not stage:
        print(f"Could not open USD file: {file_path}")
        return None


    print(f"Opened USD file: {file_path}")


    prim_data = {}
    for prim in stage.Traverse():

        prim_path = str(prim.GetPath())
        prim_type = prim.GetTypeName()
        prim_name = prim.GetName()
        print(f"Analyzing Prim: {prim_name} | Type: {prim_type} | Path: {prim_path}")


        properties = {} #use this section to see the properties of the USD
        for prop in prim.GetProperties():
            prop_name = prop.GetName()
            prop_value = prim.GetAttribute(prop_name).Get()
            print(f"Property: {prop_name} | Value: {prop_value}")

        rigid_body_api = UsdPhysics.RigidBodyAPI.Get(stage, prim_path)

        if rigid_body_api:
            print(f"The prim '{prim_path}' has RigidBodyAPI applied.")
        else:
            print(f"The prim '{prim_path}' does NOT have RigidBodyAPI applied.")
            if prim.CanApplyAPI(UsdPhysics.RigidBodyAPI):
                print("Applying RigidBodyAPI...")
                UsdPhysics.RigidBodyAPI.Apply(prim)
                print("RigidBodyAPI applied.")

        mass_api = UsdPhysics.MassAPI.Get(stage, prim_path)
        if mass_api:
            print(f"The prim '{prim_path}' has MassAPI applied.")
        else:
            print(f"The prim '{prim_path}' does NOT have MassAPI applied.")
            if prim.CanApplyAPI(UsdPhysics.MassAPI):
                print("Applying MassAPI...")
                UsdPhysics.MassAPI.Apply(prim)
                print("MassAPI applied.")


        collision_api = UsdPhysics.CollisionAPI.Get(stage, prim_path)
        if collision_api:
            print(f"The prim '{prim_path}' has CollisionAPI applied.")
        else:
            print(f"The prim '{prim_path}' does NOT have CollisionAPI applied.")
            if prim.CanApplyAPI(UsdPhysics.CollisionAPI):
                print("Applying CollisionAPI...")
                UsdPhysics.CollisionAPI.Apply(prim)
                print("CollisionAPI applied.")
        stage.GetRootLayer().Save()
        break # as we are just doing it for the root


    return prim_data


def main():

    usd_file_path = "/home/xuxuezhou/isaac-sim-assets-1/Assets/Isaac/4.5/Isaac/Props/Shapes/sphere_physics.usd"

    prim_data = analyze_usd_file(usd_file_path)



if __name__ == "__main__":
    main()